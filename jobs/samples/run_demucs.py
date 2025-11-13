#!/usr/bin/env python3
from pathlib import Path

import torch
import torchaudio
from demucs.apply import apply_model  # single‑call forward pass   :contentReference[oaicite:1]{index=1}
from demucs.audio import AudioFile
from demucs.pretrained import get_model  # pretrained weights loader :contentReference[oaicite:0]{index=0}


def run_demucs(audio_path: str, out_dir: str, model_name: str = "htdemucs", device: str = None):
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

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
    run_demucs(
        "../resources/musdb18/test/htdemucs/Al James - Schoolboy Facination.stem.track0/Al James - Schoolboy Facination.stem.track0.m4a",
        "../resources/musdb18/test/"
    )
