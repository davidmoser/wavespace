import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Literal, Optional

import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from utils.plot_cqt_comparison import load_audio_segment


def compute_log_stft(
        waveform: Tensor,
        sample_rate: int,
        *,
        n_fft: int,
        hop_length: int,
        min_frequency_hz: float,
        total_bins: int,
) -> Tensor:
    """Compute a log-magnitude STFT without normalization.

    The STFT is cropped to the requested number of frequency bins starting at
    ``min_frequency_hz``. If the requested number of bins is not available from
    the transform, the result is zero-padded to match ``total_bins``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape (channels, samples)")

    mono = waveform.mean(dim=0)
    stft = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    )
    magnitude = torch.abs(stft)
    log_magnitude = torch.log1p(magnitude)

    freq_bins_available = log_magnitude.shape[0]
    max_frequency = sample_rate / 2
    start_bin = int(math.floor(min_frequency_hz / max_frequency * (freq_bins_available - 1)))
    start_bin = max(start_bin, 0)
    end_bin = min(start_bin + total_bins, freq_bins_available)

    sliced = log_magnitude[start_bin:end_bin, :]
    if sliced.shape[0] < total_bins:
        padding = (0, 0, 0, total_bins - sliced.shape[0])
        sliced = F.pad(sliced, padding)
    return sliced


def normalize_stft_frames(
        log_stft: Tensor,
        *,
        normalization: Literal["global", "frame"],
        scale_values: float,
) -> Tensor:
    """Normalize STFT magnitudes globally or per frame."""
    if normalization not in {"global", "frame"}:
        raise ValueError("normalization must be 'global' or 'frame'")

    if normalization == "global":
        max_value = log_stft.max()
        if max_value > 0:
            normalized = log_stft / max_value
        else:
            normalized = torch.zeros_like(log_stft)
    else:
        frame_max = log_stft.max(dim=0).values
        safe_max = torch.where(frame_max > 0, frame_max, torch.ones_like(frame_max))
        normalized = log_stft / safe_max
        normalized = torch.where(frame_max > 0, normalized, torch.zeros_like(normalized))

    if scale_values != 1:
        normalized = normalized * scale_values
    return normalized.clamp(0.0, 1.0)


def stft_frame_to_column_grid(
        frame: Tensor,
        *,
        bins_per_column: int,
        total_bins: int,
) -> Tensor:
    """Arrange a single STFT frame into stacked frequency columns."""
    num_columns = math.ceil(total_bins / bins_per_column)
    grid = torch.zeros((bins_per_column, num_columns), dtype=torch.float32)
    for bin_index, value in enumerate(frame):
        column_index = bin_index // bins_per_column
        row_index = bin_index % bins_per_column
        grid[bins_per_column - 1 - row_index, column_index] = value
    return grid


def create_frames_from_stft(
        normalized_stft: Tensor,
        *,
        bins_per_column: int,
        total_bins: int,
        frame_height_pixels: int,
        column_width_pixels: int,
        colormap: str,
) -> Iterable[np.ndarray]:
    """Generate colorized frames from a normalized STFT."""
    cmap = matplotlib.colormaps.get_cmap(colormap)
    num_columns = math.ceil(total_bins / bins_per_column)
    output_width = num_columns * column_width_pixels

    for frame_idx in range(normalized_stft.shape[1]):
        grid = stft_frame_to_column_grid(
            normalized_stft[:, frame_idx],
            bins_per_column=bins_per_column,
            total_bins=total_bins,
        )
        resized = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=(frame_height_pixels, output_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        rgba = cmap(resized.cpu().numpy())
        yield (rgba[..., :3] * 255).astype(np.uint8)


def create_stft_video(
        audio_file: str,
        output_video: str,
        duration_seconds: Optional[float],
        scale_values: float = 1,
        stft_bins: int = 512,
        stft_bins_per_column: int = 32,
        stft_frames_per_s: int = 70,
        stft_fmin: float = 0.0,
        stft_vertical_pixels: int = 128,
        column_width_pixels: int = 32,
        normalization: Literal["global", "frame"] = "global",
        colormap: str = "viridis",
) -> None:
    """Create a video where each frame is an STFT time slice split into columns."""
    waveform, sample_rate, _ = load_audio_segment(audio_file, duration=duration_seconds)
    hop_length = sample_rate // stft_frames_per_s
    n_fft = max(stft_bins * 2, 2)
    log_stft = compute_log_stft(
        waveform,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        min_frequency_hz=stft_fmin,
        total_bins=stft_bins,
    )
    normalized = normalize_stft_frames(
        log_stft,
        normalization=normalization,
        scale_values=scale_values,
    )

    frames = list(create_frames_from_stft(
        normalized,
        bins_per_column=stft_bins_per_column,
        total_bins=stft_bins,
        frame_height_pixels=stft_vertical_pixels,
        column_width_pixels=column_width_pixels,
        colormap=colormap,
    ))

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_without_audio = temp_dir_path / "video.mp4"
        audio_path = temp_dir_path / "audio.wav"

        iio.imwrite(video_without_audio, frames, fps=stft_frames_per_s)
        torchaudio.save(audio_path, waveform, sample_rate)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_without_audio),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to mux audio into video: "
                f"{result.stderr.strip() or 'unknown ffmpeg error.'}"
            )
