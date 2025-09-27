"""Encodec roundtrip script.

This script loads an audio file, encodes it with the Encodec codec from Meta,
then decodes it and writes the reconstructed waveform next to the original
file. The output filename contains the chosen quality level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


QUALITY_LEVELS = [1.5, 3.0, 6.0, 12.0]


def _load_model(model_name: str, device: str) -> EncodecModel:
    if model_name == "48khz":
        model = EncodecModel.encodec_model_48khz()
    else:
        model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidths(QUALITY_LEVELS)
    model.to(device)
    model.set_device(device)
    return model


def encodec_roundtrip(
    input_path: Path,
    *,
    quality: float = 6.0,
    model_name: str = "24khz",
    output_format: str = "wav",
    device: Optional[str] = None,
) -> Path:
    """Encode and decode an audio file with Encodec.

    Args:
        input_path: Path to the source audio file to round-trip.
        quality: Target bandwidth (kbps) for Encodec. Must be in QUALITY_LEVELS.
        model_name: Which pretrained Encodec model to use ("24khz" or "48khz").
        output_format: Audio format for the reconstructed file ("wav" or "mp3").
        device: Optional device override. Defaults to CUDA when available.

    Returns:
        The path where the reconstructed audio was written.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if quality not in QUALITY_LEVELS:
        raise ValueError(
            f"quality must be one of {QUALITY_LEVELS}; received {quality!r}"
        )

    if output_format not in {"wav", "mp3"}:
        raise ValueError("output_format must be either 'wav' or 'mp3'")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    waveform, sample_rate = torchaudio.load(str(input_path))

    model = _load_model(model_name, device)
    model.eval()

    # Prepare audio for the model
    waveform = waveform.to(device)
    audio = convert_audio(
        waveform,
        sample_rate,
        model.sample_rate,
        model.channels,
    ).unsqueeze(0)

    with torch.inference_mode():
        encoded = model.encode(audio, target_bandwidth=quality)
        decoded = model.decode(encoded)

    # Convert back to the original sample rate and channel count
    decoded = convert_audio(
        decoded.squeeze(0).cpu(),
        model.sample_rate,
        sample_rate,
        waveform.shape[0],
    )

    suffix = f"_encodec_{quality}"
    output_path = input_path.with_name(
        f"{input_path.stem}{suffix}.{output_format}"
    )

    if output_format == "wav":
        torchaudio.save(str(output_path), decoded, sample_rate)
    else:
        torchaudio.save(str(output_path), decoded, sample_rate, format="mp3")

    return output_path


if __name__ == "__main__":
    INPUT_FILE = Path("path/to/audio.wav")
    ROUNDTRIP_QUALITY = 6.0
    MODEL_NAME = "24khz"
    OUTPUT_FORMAT = "wav"
    DEVICE_OVERRIDE: Optional[str] = None

    if INPUT_FILE.exists():
        result_path = encodec_roundtrip(
            INPUT_FILE,
            quality=ROUNDTRIP_QUALITY,
            model_name=MODEL_NAME,
            output_format=OUTPUT_FORMAT,
            device=DEVICE_OVERRIDE,
        )
        print(f"Saved decoded audio to {result_path}")
    else:
        print(
            "Update INPUT_FILE to point to an existing audio file before running the "
            "roundtrip."
        )
