"""Run pitch detection models on audio files and export piano-roll visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor

from pitch_detection_supervised.train import create_model


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model_name: Optional[str] = None,
    model_config: Optional[dict] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Load a supervised pitch detection model from a checkpoint."""
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict: dict = payload.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint payload missing 'model_state_dict'.")

    config_dict = payload.get("config_dict")
    payload_model_name = payload.get("model_class")

    resolved_model_name = model_name or (
        config_dict.get("model_name") if isinstance(config_dict, dict) else None
    ) or payload_model_name
    if resolved_model_name is None:
        raise KeyError("Unable to determine model name from checkpoint. Provide model_name explicitly.")

    resolved_model_config: dict = {}
    if isinstance(config_dict, dict):
        cfg = config_dict.get("model_config")
        if isinstance(cfg, dict):
            resolved_model_config.update(cfg)
    if isinstance(model_config, dict):
        resolved_model_config.update(model_config)

    model = create_model(resolved_model_name, resolved_model_config)
    model.load_state_dict(state_dict)
    return model, payload


def load_audio_segment(
    audio_path: Path,
    *,
    duration: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, int]:
    """Load an audio file and optionally crop it to a duration in seconds."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.to(torch.float32)
    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive when provided.")
        max_samples = int(round(duration * sample_rate))
        waveform = waveform[..., :max_samples]
    if device is not None:
        waveform = waveform.to(device)
    return waveform, sample_rate


def resample_for_encodec(
    waveform: Tensor,
    original_sample_rate: int,
    encoder: EncodecModel,
    *,
    target_duration: Optional[float] = None,
) -> Tuple[Tensor, int]:
    """Resample a waveform tensor to match the Encodec encoder expectations."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape (channels, samples)")

    resampled = convert_audio(
        waveform,
        original_sample_rate,
        encoder.sample_rate,
        encoder.channels,
    )
    if target_duration is not None:
        max_samples = int(round(target_duration * encoder.sample_rate))
        resampled = resampled[..., :max_samples]
    return resampled, encoder.sample_rate


def chunk_waveform(
    waveform: Tensor,
    chunk_samples: int,
) -> Iterable[Tuple[Tensor, int]]:
    """Yield chunks of the waveform padded to ``chunk_samples`` samples."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    total_samples = waveform.shape[-1]
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[..., start:end]
        valid_samples = chunk.shape[-1]
        if valid_samples < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - valid_samples))
        yield chunk, valid_samples


def encode_chunks_to_latents(
    waveform: Tensor,
    encoder: EncodecModel,
    chunk_samples: int,
    *,
    device: torch.device,
) -> Tuple[List[Tensor], List[int]]:
    """Convert audio chunks to Encodec pre-quant latents."""
    latents: List[Tensor] = []
    valid_samples: List[int] = []
    encoder_device = next(encoder.parameters()).device
    with torch.inference_mode():
        for chunk, valid in chunk_waveform(waveform, chunk_samples):
            valid_samples.append(valid)
            chunk_batch = chunk.unsqueeze(0).to(encoder_device)
            latent = encoder.encoder(chunk_batch).squeeze(0).to(device)
            latents.append(latent.to(dtype=torch.float32).contiguous())
    return latents, valid_samples


def run_model_on_latents(
    model: torch.nn.Module,
    latents: Sequence[Tensor],
    valid_samples: Sequence[int],
    chunk_samples: int,
) -> Tensor:
    """Run the pitch detection model on latent chunks and stitch predictions."""
    if len(latents) != len(valid_samples):
        raise ValueError("latents and valid_samples must have the same length")

    device = next(model.parameters()).device
    model.eval()
    predictions: List[Tensor] = []
    with torch.inference_mode():
        for latent, valid in zip(latents, valid_samples):
            inputs = latent.unsqueeze(0).to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu()
            total_frames = probs.shape[-1]
            frames = total_frames
            if valid < chunk_samples:
                frames = max(1, int(round((valid / chunk_samples) * total_frames)))
                probs = probs[..., :frames]
            predictions.append(probs)
    concatenated = torch.cat(predictions, dim=-1)
    return concatenated


def create_prediction_image(
    prediction: Tensor,
    *,
    duration_seconds: float,
    pixels_per_second: int = 10,
    vertical_pixels: int = 256,
) -> np.ndarray:
    """Convert model predictions into a colored piano-roll style image."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if pixels_per_second <= 0:
        raise ValueError("pixels_per_second must be positive")
    if vertical_pixels <= 0:
        raise ValueError("vertical_pixels must be positive")

    freq_bins, total_frames = prediction.shape
    if total_frames == 0:
        raise ValueError("Prediction tensor has zero time frames.")

    width = max(1, int(math.ceil(duration_seconds * pixels_per_second)))
    resized = F.interpolate(
        prediction.unsqueeze(0).unsqueeze(0),
        size=(vertical_pixels, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    clipped = torch.clamp(resized, 0.0, 1.0).numpy()
    rgba = cm.get_cmap("viridis")(clipped)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def save_image(array: np.ndarray, output_path: Path) -> None:
    """Save a numpy array as a PNG image."""
    image = Image.fromarray(array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def run_inference(
    *,
    checkpoint_path: Path,
    audio_path: Path,
    output_image_path: Path,
    chunk_duration: float = 2.0,
    analysis_duration: Optional[float] = None,
    pixels_per_second: int = 10,
    vertical_pixels: int = 256,
    model_name: Optional[str] = None,
    model_config: Optional[dict] = None,
    device: Optional[str] = None,
    encoder_bandwidth: float = 24.0,
) -> Tensor:
    """Run a pitch detection model on an audio file and save a visualization."""
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    model_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, _ = load_checkpoint(
        checkpoint_path,
        model_name=model_name,
        model_config=model_config,
        map_location=model_device,
    )
    model.to(model_device)
    model.eval()

    encoder = EncodecModel.encodec_model_24khz()
    encoder.set_target_bandwidth(encoder_bandwidth)
    encoder = encoder.to(model_device)
    encoder.eval()

    waveform, original_sample_rate = load_audio_segment(
        audio_path,
        duration=analysis_duration,
    )
    resampled, encoder_sample_rate = resample_for_encodec(
        waveform,
        original_sample_rate,
        encoder,
        target_duration=analysis_duration,
    )
    if resampled.shape[-1] == 0:
        raise ValueError("The requested audio segment is empty after resampling.")

    chunk_samples = int(round(chunk_duration * encoder_sample_rate))
    latents, valid_samples = encode_chunks_to_latents(
        resampled,
        encoder,
        chunk_samples,
        device=model_device,
    )
    prediction = run_model_on_latents(
        model,
        latents,
        valid_samples,
        chunk_samples,
    )
    total_valid_samples = sum(valid_samples)
    duration_seconds = total_valid_samples / encoder_sample_rate
    image_array = create_prediction_image(
        prediction,
        duration_seconds=duration_seconds,
        pixels_per_second=pixels_per_second,
        vertical_pixels=vertical_pixels,
    )
    save_image(image_array, output_image_path)
    return prediction


if __name__ == "__main__":
    # Update these paths before running the script directly.
    checkpoint = Path("path/to/checkpoint.pt")
    audio_file = Path("path/to/audio.mp3")
    output_image = Path("path/to/output.png")

    if checkpoint.exists() and audio_file.exists():
        run_inference(
            checkpoint_path=checkpoint,
            audio_path=audio_file,
            output_image_path=output_image,
        )
    else:
        raise SystemExit(
            "Please edit runners/pitch_detection_inference.py with valid paths before executing."
        )
