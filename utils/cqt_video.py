import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Literal, Optional

import imageio.v3 as iio
import librosa
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from utils.plot_cqt_comparison import load_audio_segment


def compute_log_cqt(
        waveform: Tensor,
        sample_rate: int,
        *,
        n_bins: int,
        bins_per_octave: int,
        hop_length: int,
        fmin: float,
) -> Tensor:
    """Compute the log-magnitude CQT without normalization."""
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
    return torch.from_numpy(log_magnitude.astype(np.float32))


def normalize_cqt_frames(
        log_cqt: Tensor,
        *,
        normalization: Literal["global", "frame"],
        scale_values: float,
) -> Tensor:
    """Normalize CQT magnitudes globally or per frame."""
    if normalization not in {"global", "frame"}:
        raise ValueError("normalization must be 'global' or 'frame'")

    if normalization == "global":
        max_value = log_cqt.max()
        if max_value > 0:
            normalized = log_cqt / max_value
        else:
            normalized = torch.zeros_like(log_cqt)
    else:
        frame_max = log_cqt.max(dim=0).values
        safe_max = torch.where(frame_max > 0, frame_max, torch.ones_like(frame_max))
        normalized = log_cqt / safe_max
        normalized = torch.where(frame_max > 0, normalized, torch.zeros_like(normalized))

    if scale_values != 1:
        normalized = normalized * scale_values
    return normalized.clamp(0.0, 1.0)


def cqt_frame_to_octave_grid(
        frame: Tensor,
        *,
        bins_per_octave: int,
        total_bins: int,
) -> Tensor:
    """Rearrange a single CQT frame into octave columns."""
    num_columns = math.ceil(total_bins / bins_per_octave)
    grid = torch.zeros((bins_per_octave, num_columns), dtype=torch.float32)
    for bin_index, value in enumerate(frame):
        octave_index = bin_index // bins_per_octave
        note_index = bin_index % bins_per_octave
        # Flip vertically so lower notes are at the bottom.
        grid[bins_per_octave - 1 - note_index, octave_index] = value
    return grid


def create_frames_from_cqt(
        normalized_cqt: Tensor,
        *,
        bins_per_octave: int,
        total_bins: int,
        frame_height_pixels: int,
        column_width_pixels: int,
        colormap: str,
) -> Iterable[np.ndarray]:
    """Generate colorized frames from a normalized CQT."""
    cmap = matplotlib.colormaps.get_cmap(colormap)
    num_columns = math.ceil(total_bins / bins_per_octave)
    output_width = num_columns * column_width_pixels

    for frame_idx in range(normalized_cqt.shape[1]):
        grid = cqt_frame_to_octave_grid(
            normalized_cqt[:, frame_idx],
            bins_per_octave=bins_per_octave,
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


def create_cqt_video(
        audio_file: str,
        output_video: str,
        duration_seconds: Optional[float],
        scale_values: float = 1,
        cqt_bins: int = 84,
        cqt_bins_per_octave: int = 12,
        cqt_frames_per_s: int = 70,
        cqt_fmin: float = 32.7,
        cqt_vertical_pixels: int = 128,
        column_width_pixels: int = 32,
        normalization: Literal["global", "frame"] = "global",
        colormap: str = "viridis",
) -> None:
    """Create a video where each frame is a CQT time slice arranged by octave."""
    waveform, sample_rate, _ = load_audio_segment(audio_file, duration=duration_seconds)
    hop_length = sample_rate // cqt_frames_per_s
    log_cqt = compute_log_cqt(
        waveform,
        sample_rate,
        n_bins=cqt_bins,
        bins_per_octave=cqt_bins_per_octave,
        hop_length=hop_length,
        fmin=cqt_fmin,
    )
    normalized = normalize_cqt_frames(
        log_cqt,
        normalization=normalization,
        scale_values=scale_values,
    )

    frames = list(create_frames_from_cqt(
        normalized,
        bins_per_octave=cqt_bins_per_octave,
        total_bins=cqt_bins,
        frame_height_pixels=cqt_vertical_pixels,
        column_width_pixels=column_width_pixels,
        colormap=colormap,
    ))

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_without_audio = temp_dir_path / "video.mp4"
        audio_path = temp_dir_path / "audio.wav"

        iio.imwrite(video_without_audio, frames, fps=cqt_frames_per_s)
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
