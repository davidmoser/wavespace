import math
import os

import matplotlib.pyplot as plt
import torch
import torchaudio


def pitch_heatmap(
        audio_path: str,
        N: int = 200,  # number of k-offsets
        n_fft: int = 4196,
        hop_length: int = 200,
        max_frames: int = 1_000_000,
        target_sr: int = 44_100):
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(0, keepdim=True)[:, :max_frames]

    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop_length, window=window, return_complex=True).abs().squeeze(0)
    F, T = spec.shape

    ks = torch.logspace(0, torch.log10(torch.tensor(200)), steps=N, dtype=torch.double, device=spec.device)

    out = torch.zeros((ks.numel(), T), dtype=spec.dtype, device=spec.device)
    for i, k in enumerate(ks):
        # maximum integer multiple m so that round(m·k) is still < F
        m_max = math.floor((F - 1) / k.item())
        m = torch.arange(1, m_max + 1, device=spec.device)
        idx = torch.round(m * k).to(torch.long)  # integer frequency bins
        out[i] = spec[idx].sum(dim=0) / m_max

    # mark the maxima (summed over three steps)
    const_val   = out.max().item()                     # as before

    wsum       = torch.empty_like(out)
    wsum[0]    = out[0]      + out[1]
    wsum[-1]   = out[-2]     + out[-1]
    wsum[1:-1] = out[:-2] + out[1:-1] + out[2:]
    col_max_idx  = wsum.argmax(dim=0, keepdim=True)    # (1, T)

    cutoff       = const_val / 2
    col_max_vals = wsum.max(dim=0).values              # strength of each column peak
    valid        = col_max_vals >= cutoff              # (T,) mask

    if valid.any():                                    # write only strong peaks
        rows = col_max_idx[0, valid]                   # winning row per valid column
        cols = torch.nonzero(valid, as_tuple=False).squeeze(1)
        out[rows, cols] = const_val

    # plot
    fig_w, fig_h = T / 300, N / 300
    plt.figure(figsize=(fig_w, fig_h), dpi=300)
    plt.imshow(out.cpu(), origin='lower', aspect='auto',
               interpolation='nearest')
    plt.axis('off')
    out_path = os.path.splitext(audio_path)[0] + "_pitch_heat.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
