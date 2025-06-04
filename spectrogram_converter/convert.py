import math
import pathlib

import torch
import torchaudio
from torch.utils.data import DataLoader

from spectrogram_converter.audio_folder import AudioFolder
from spectrogram_converter.configuration import Configuration


def convert(cfg: Configuration) -> None:
    dev = cfg.resolved_device
    print(f"Using device: {dev}")

    ds = AudioFolder(pathlib.Path(cfg.audio_dir), cfg.sr, cfg.dur)
    loader = DataLoader(ds, batch_size=cfg.batch,
                        shuffle=True, pin_memory=True,
                        num_workers=cfg.num_workers)

    if cfg.type == "mel":
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=4096, hop_length=512,
            n_mels=256, power=2.0).to(dev)
    elif cfg.type == "log":
        spec_fft = torchaudio.transforms.Spectrogram(n_fft=4096, hop_length=512, power=2.0).to(dev)
        W = calculate_log_matrix(n_fft=4096, sr=cfg.sr, log_bins=256)

        def spec(x):
            return torch.matmul(W, spec_fft(x))
    else:
        raise TypeError(f"Unknown type {cfg.type}")

    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(dev)

    print(f"Converting: {len(ds)} samples")

    specs = []
    for wav in loader:  # (B,T)
        wav = wav.to(dev)
        x = to_db(spec(wav))  # (B,F,T)
        specs.append(x.to(torch.float16))

    specs = torch.cat(specs, dim=0)

    print(f"Saving specs to {cfg.spec_file}")
    torch.save(specs, cfg.spec_file)

    print("Done")


def calculate_log_matrix(n_fft, sr, log_bins) -> torch.Tensor:
    n_bins = n_fft // 2 + 1
    lower_k = 50 * n_fft // sr  # start at 50 Hz
    upper_k = n_bins - 1  # highest freq of this FFT

    edges = torch.exp(torch.linspace(math.log(lower_k), math.log(upper_k), steps=log_bins + 2))

    min_width = 0.5  # ≥½ bin avoids degenerate triangles
    edges[1:] = torch.maximum(edges[1:], edges[:-1] + min_width)

    bins = torch.arange(n_bins, device=edges.device)[None, :].float()
    l, c, r = edges[:-2, None], edges[1:-1, None], edges[2:, None]

    den_l = (c - l).clamp_min(1e-6)  # ε-clamp
    den_r = (r - c).clamp_min(1e-6)

    left = ((bins - l) / den_l).clamp(0, 1) * (bins <= c)
    right = ((r - bins) / den_r).clamp(0, 1) * (bins >= c)
    W = left + right  # (B,F)

    # fix any rows whose area stayed zero
    row_sum = W.sum(1, keepdim=True)
    deg = row_sum.squeeze() == 0
    if deg.any():
        centre_idx = c[deg, 0].round().long().clamp_max(n_bins - 1)
        W[deg, :] = 0.0
        W[deg, centre_idx] = 1.0
        row_sum = W.sum(1, keepdim=True)  # now non-zero

    W /= row_sum
    return W
