from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import librosa
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch import Tensor


def load_audio_segment(
        audio_file: str,
        *,
        duration: Optional[float] = None,
        device: Optional[torch.device] = None,
) -> Tuple[Tensor, int, float]:
    """Load an audio file and optionally crop it to a duration in seconds."""
    audio_path = Path(audio_file)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.to(torch.float32)
    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive when provided.")
        max_samples = int(round(duration * sample_rate))
        waveform = waveform[..., :max_samples]
        eff_duration = duration
    else:
        eff_duration = waveform.shape[1] / sample_rate
    if device is not None:
        waveform = waveform.to(device)
    return waveform, sample_rate, eff_duration


def compute_cqt_representation(
        waveform: Tensor,
        sample_rate: int,
        *,
        n_bins: int,
        bins_per_octave: int,
        hop_length: int,
        fmin: float,
) -> Tensor:
    """Compute a normalized CQT magnitude representation for visualization."""
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape (channels, samples)")

    mono = waveform.mean(dim=0)
    mono_np = mono.to(torch.float32).cpu().numpy()
    cqt = librosa.cqt(
        mono_np,
        sr=sample_rate,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        fmin=fmin,
    )
    magnitude = np.abs(cqt)
    log_magnitude = np.log1p(magnitude)
    log_magnitude = log_magnitude - log_magnitude.min()
    max_value = log_magnitude.max()
    if max_value > 0:
        normalized = log_magnitude / max_value
    else:
        normalized = np.zeros_like(log_magnitude)
    return torch.from_numpy(normalized.astype(np.float32))


def create_prediction_image(
        prediction: Tensor,
        cqt_representation: Tensor,
        *,
        duration_seconds: float,
        scale_values: float,
        pixels_per_second: int = 50,
        vertical_pixels: int = 128,
        cqt_vertical_pixels: int = 128,
        separator_height: int = 2,
        colormap: str = "viridis",
) -> np.ndarray:
    """Convert predictions and CQT data into a stacked visualization."""
    if separator_height < 0:
        raise ValueError("separator_height must be non-negative")
    if pixels_per_second <= 0:
        raise ValueError("pixels_per_second must be positive")
    if vertical_pixels <= 0 or cqt_vertical_pixels <= 0:
        raise ValueError("vertical pixel counts must be positive")
    width = max(1, int(math.ceil(duration_seconds * pixels_per_second)))

    prediction_resized = F.interpolate(
        prediction.unsqueeze(0).unsqueeze(0),
        size=(vertical_pixels, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    prediction_scaled = (prediction_resized * scale_values).clamp(0.0, 1.0)

    cqt_resized = F.interpolate(
        cqt_representation.unsqueeze(0).unsqueeze(0),
        size=(cqt_vertical_pixels, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    cqt_scaled = cqt_resized.clamp(0.0, 1.0)

    if separator_height == 0:
        stacked = torch.cat((prediction_scaled, cqt_scaled), dim=0)
    else:
        separator = torch.ones((separator_height, width), dtype=torch.float32)
        stacked = torch.cat((prediction_scaled, separator, cqt_scaled), dim=0)

    rgba = matplotlib.colormaps.get_cmap(colormap)(stacked.cpu().numpy())
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def save_image(array: np.ndarray, output: str) -> None:
    """Save a numpy array as a PNG image."""
    image = Image.fromarray(array)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def plot_cqt_comparison(
        audio_file: str,
        prediction: Tensor,
        output_image: str,
        duration_seconds: Optional[float],
        scale_values: float = 1,
        prediction_vertical_pixels: int = 128,
        cqt_bins: int = 84,
        cqt_bins_per_octave: int = 12,
        cqt_hop_length: int = 256,
        cqt_fmin: float = 32.7,
        cqt_vertical_pixels: int = 128,
        separator_height: int = 2,
        pixels_per_second: int = 50,
        colormap: str = "viridis",
):
    waveform, sample_rate, eff_duration = load_audio_segment(audio_file, duration=duration_seconds)
    cqt_representation = compute_cqt_representation(
        waveform,
        sample_rate,
        n_bins=cqt_bins,
        bins_per_octave=cqt_bins_per_octave,
        hop_length=cqt_hop_length,
        fmin=cqt_fmin,
    )
    image_array = create_prediction_image(
        prediction,
        cqt_representation,
        scale_values=scale_values,
        duration_seconds=eff_duration,
        pixels_per_second=pixels_per_second,
        vertical_pixels=prediction_vertical_pixels,
        cqt_vertical_pixels=cqt_vertical_pixels,
        separator_height=separator_height,
        colormap=colormap,
    )
    save_image(image_array, output_image)
