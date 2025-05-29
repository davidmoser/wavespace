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

    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr, n_fft=4096, hop_length=512,
        n_mels=256, power=2.0).to(dev)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(dev)

    num_samples = len(loader) * cfg.batch
    print(f"Converting: {num_samples} samples")

    specs = []
    for wav in loader:  # (B,T)
        wav = wav.to(dev)
        x = to_db(spec(wav))  # (B,F,T)
        specs.append(x.to(torch.float16))

    specs = torch.cat(specs, dim=0)

    print(f"Saving specs to {cfg.spec_file}")
    torch.save(specs, cfg.spec_file)

    print("Done")
