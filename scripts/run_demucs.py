#!/usr/bin/env python3
"""
separate_stems.py – minimal Demucs‑v4 example
---------------------------------------------
$ python separate_stems.py my_track.mp3            # CPU, default model (htdemucs)
$ python separate_stems.py my_track.mp3 --model mdx_q --device cuda
Results land in  ./separated/<model>/<track>/bass.wav, drums.wav, other.wav, vocals.wav
"""
import argparse
from pathlib import Path

import torch
import torchaudio
from demucs.apply import apply_model  # single‑call forward pass   :contentReference[oaicite:1]{index=1}
from demucs.audio import AudioFile
from demucs.pretrained import get_model  # pretrained weights loader :contentReference[oaicite:0]{index=0}


def separate(audio_path: Path, model_name: str, device: str, out_dir: Path):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device).eval()

    wav = AudioFile(audio_path).read(streams=[0],
                                              samplerate=model.samplerate,
                                              channels=model.audio_channels)

    mix = wav.to(device)
    sources = apply_model(model, mix, device=device)[0]

    target_dir = out_dir / model_name / audio_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    for tensor, name in zip(sources, model.sources):
        torchaudio.save(str(target_dir / f"{name}.wav"), tensor.cpu(), model.samplerate)
        print(f"✓ {target_dir / name}.wav")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="input file (mp3/wav/flac…)")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--out", default="separated")
    args = p.parse_args()
    separate(Path(args.audio), args.model, args.device, Path(args.out))
