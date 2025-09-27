"""Utility to generate and save a sine wave audio clip."""

from pathlib import Path
from typing import Union

import math
import wave

import numpy as np


def create_sine_audio(
    frequency: float = 14000.0,
    duration: float = 1.0,
    volume: float = 1.0,
    sample_rate: int = 48_000,
    output_folder: Union[str, Path] = "../resources/sine/",
) -> Path:
    if not 0 <= volume <= 1:
        raise ValueError("volume must be between 0 and 1 inclusive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    output_folder = Path(output_folder)
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    sr_khz = sample_rate / 1000
    filename = f"sine_{frequency:g}Hz_{duration:g}s_{sr_khz:g}kHz.wav"
    output_path = output_folder / filename

    total_samples = int(round(duration * sample_rate))
    times = np.linspace(0, duration, num=total_samples, endpoint=False)
    waveform = np.sin(2 * math.pi * frequency * times)

    max_amplitude = np.iinfo(np.int16).max
    scaled_waveform = (waveform * volume * max_amplitude).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(scaled_waveform.tobytes())

    return output_path


if __name__ == "__main__":
    create_sine_audio()
