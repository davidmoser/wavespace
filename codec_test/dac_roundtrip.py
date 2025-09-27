"""DAC roundtrip script.

This script loads an audio file, encodes it with the Descript Audio Codec (DAC),
then decodes it and writes the reconstructed waveform next to the original
file. The output filename contains the chosen model configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torchaudio
from audiotools import AudioSignal
from dac.utils import load_model

MODEL_TYPES = ("16khz", "24khz", "44khz")
MODEL_BITRATES = ("8kbps", "16kbps")


def _load_model(
    model_type: str,
    model_bitrate: str,
    device: str,
    *,
    model_tag: str = "latest",
):
    model = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
    )
    model.to(device)
    model.eval()
    return model


def dac_roundtrip(
    input_path: Path,
    *,
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    model_tag: str = "latest",
    output_format: str = "wav",
    device: Optional[str] = None,
    win_duration: float = 1.0,
    normalize_db: Optional[float] = -16.0,
) -> Path:
    """Encode and decode an audio file with the Descript Audio Codec (DAC).

    Args:
        input_path: Path to the source audio file to round-trip.
        model_type: Which pretrained DAC model to use ("16khz", "24khz", or "44khz").
        model_bitrate: Target bitrate for DAC ("8kbps" or "16kbps").
        model_tag: Specific model version tag to download. Defaults to "latest".
        output_format: Audio format for the reconstructed file ("wav" or "mp3").
        device: Optional device override. Defaults to CUDA when available.
        win_duration: Window duration (seconds) for chunked compression.
        normalize_db: Target loudness for normalization before encoding.

    Returns:
        The path where the reconstructed audio was written.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {MODEL_TYPES}; received {model_type!r}"
        )

    if model_bitrate not in MODEL_BITRATES:
        raise ValueError(
            f"model_bitrate must be one of {MODEL_BITRATES}; received {model_bitrate!r}"
        )

    if model_bitrate == "16kbps" and model_type != "44khz":
        raise ValueError("The 16kbps model is only available for the 44khz variant.")

    if output_format not in {"wav", "mp3"}:
        raise ValueError("output_format must be either 'wav' or 'mp3'")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    signal = AudioSignal(str(input_path))
    signal = signal.to(device)

    model = _load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        model_tag=model_tag,
        device=device,
    )

    dac_file = model.compress(
        signal,
        win_duration=win_duration,
        normalize_db=normalize_db,
    )
    reconstructed = model.decompress(dac_file)
    reconstructed = reconstructed.cpu()

    waveform = reconstructed.audio_data.squeeze(0)
    sample_rate = reconstructed.sample_rate

    suffix = f"_dac_{model_type}_{model_bitrate}"
    output_path = input_path.with_name(
        f"{input_path.stem}{suffix}.{output_format}"
    )

    if output_format == "wav":
        torchaudio.save(str(output_path), waveform, sample_rate)
    else:
        torchaudio.save(str(output_path), waveform, sample_rate, format="mp3")

    return output_path


if __name__ == "__main__":
    input_file = Path("../resources/sine/sine_1000Hz_1s_16kHz.wav")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    result_path = dac_roundtrip(
        input_file,
        model_type="16khz",
        model_bitrate="8kbps",
        output_format="wav",
        device=None,
    )
    print(f"Saved decoded audio to {result_path}")
