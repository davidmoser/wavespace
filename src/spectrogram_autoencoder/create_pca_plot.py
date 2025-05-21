#!/usr/bin/env python3
"""
latent_pca.py

Compute bottleneck-latent vectors for every *.wav in a folder, run PCA, and
plot the first two components coloured by instrument.

Example:
    python latent_pca.py --data_dir ./wav_folder \
                         --labels_csv ./labels.csv \
                         --ckpt ./spec_auto_v2.pt \
                         --out_plot latent_pca.png
"""
import argparse
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.decomposition import PCA

from spec_auto_v3 import SpecAutoNet  # same model file as before

SR, N_FFT, HOP, N_MELS = 22_050, 4096, 512, 256
spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
    n_mels=N_MELS, power=2.0
)
to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


def crop_to_multiple(x: torch.Tensor, m: int = 8, dims=(-2, -1)):
    sl = [slice(None)] * x.ndim
    for d in dims:
        sl[d] = slice(0, x.shape[d] - (x.shape[d] % m))
    return x[tuple(sl)]


def wav_to_mel(path: pathlib.Path) -> torch.Tensor:
    wav, sr0 = torchaudio.load(path)
    wav = wav.mean(0, keepdim=True)
    if sr0 != SR:
        wav = torchaudio.functional.resample(wav, sr0, SR)
    wav = wav[..., : int(20 * SR)]
    with torch.no_grad():
        mel = to_db(spec(wav))
    return crop_to_multiple(mel, 16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=pathlib.Path, required=True)
    p.add_argument("--labels_csv", type=pathlib.Path, required=True)
    p.add_argument("--ckpt", type=pathlib.Path,
                   default="../../resources/checkpoints/spec_auto_v3_brass.pt")
    p.add_argument("--out_plot", type=pathlib.Path, default="latent_pca.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--include_pc34", default=False)
    args = p.parse_args()

    # load label map: uuid4 -> instrument
    df = pd.read_csv(args.labels_csv)
    label_for_uuid = dict(zip(df["uuid4"], df["instrument"]))

    # load model
    model = SpecAutoNet().to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    # choose ≤ 1000 random .wav files, skipping dot-prefixed names
    wav_files = [p for p in args.data_dir.glob("*.wav") if not p.name.startswith('.')]
    if not wav_files:
        sys.exit("no .wav files found in data_dir")
    if len(wav_files) > 3000:
        wav_files = random.sample(wav_files, k=3000)
    if not wav_files:
        sys.exit("no .wav files found in data_dir")

    latents, labels = [], []
    for wav_path in wav_files:
        uuid = wav_path.stem.split("_")[-1]
        instr = label_for_uuid.get(uuid, None)
        if instr is None:
            print(f"[warn] uuid {uuid} missing in CSV – skipped")
            continue

        mel = wav_to_mel(wav_path).unsqueeze(0).to(args.device)  # (1,1,F,T)
        with torch.no_grad():
            _, z = model(mel, return_latent=True)
        z = z.flatten().cpu().numpy()
        latents.append(z)
        labels.append(instr)

    if len(latents) < 2:
        sys.exit("need at least two valid samples for PCA")

    X = np.vstack(latents)
    pca = PCA(n_components=4)
    comps = pca.fit_transform(X)
    print(f"explained variance: PC-1 {pca.explained_variance_ratio_[0]:.2%}, "
          f"PC-2 {pca.explained_variance_ratio_[1]:.2%}")

    # colour map: one colour per instrument
    uniq_instr = sorted(set(labels))
    color_cycle = plt.cm.get_cmap("tab10", len(uniq_instr))
    col_for_instr = {inst: color_cycle(i) for i, inst in enumerate(uniq_instr)}

    if args.include_pc34:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    # helper to plot any (i,j) PC-pair
    def plot_pair(ax, i, j, title):
        for inst in uniq_instr:
            idx = [k for k, lbl in enumerate(labels) if lbl == inst]
            ax.scatter(comps[idx, i], comps[idx, j],
                       s=5, alpha=0.8, color=col_for_instr[inst], label=inst)
        ax.set_xlabel(f"PC-{i + 1}")
        ax.set_ylabel(f"PC-{j + 1}")
        ax.set_title(title)

    if args.include_pc34:
        plot_pair(axes[0], 0, 1, "PC-1 vs PC-2")
        plot_pair(axes[1], 2, 3, "PC-3 vs PC-4")
        handles, labels_ = axes[0].get_legend_handles_labels()
    else:
        plot_pair(axes, 0, 1, "PC-1 vs PC-2")
        handles, labels_ = axes.get_legend_handles_labels()

    fig.legend(handles, labels_, frameon=False,
               bbox_to_anchor=(1.02, 0.5), loc="center left")

    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150, bbox_inches="tight")

    print(f"saved {args.out_plot}")


if __name__ == "__main__":
    main()
