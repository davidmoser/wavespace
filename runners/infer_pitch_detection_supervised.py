"""Run pitch detection models on audio files and export piano-roll visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
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
        map_location: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load a supervised pitch detection model from a checkpoint."""
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict: dict = payload.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint payload missing 'model_state_dict'.")

    config_dict = payload.get("config_dict")
    model_name = config_dict.get("model_name")
    model_config = config_dict.get("model_config")

    model = create_model(model_name, model_config)
    model.load_state_dict(state_dict)
    return model


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
        scale_values: float,
        pixels_per_second: int = 50,
        vertical_pixels: int = 128,
) -> np.ndarray:
    """Convert model predictions into a colored piano-roll style image."""
    width = max(1, int(math.ceil(duration_seconds * pixels_per_second)))
    resized = F.interpolate(
        prediction.unsqueeze(0).unsqueeze(0),
        size=(vertical_pixels, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    resized_scaled = resized * scale_values
    rgba = matplotlib.colormaps.get_cmap("viridis")(resized_scaled)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def save_image(array: np.ndarray, output_path: Path) -> None:
    """Save a numpy array as a PNG image."""
    image = Image.fromarray(array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def run_inference(
        *,
        checkpoint: str,
        audio_file: str,
        output_image: str,
        scale_values=2,
        chunk_duration: float = 2.0,
        analysis_duration: Optional[float] = None,
        device: Optional[str] = None,
        encoder_bandwidth: float = 24.0,
) -> Tensor:
    """Run a pitch detection model on an audio file and save a visualization."""
    checkpoint_path = Path(checkpoint)
    audio_file_path = Path(audio_file)
    output_image_path = Path(output_image)

    if not checkpoint_path.exists():
        raise SystemError(f"Checkpoint does not exist {checkpoint}")
    if not audio_file_path.exists():
        raise SystemError(f"Audio file does not exist {audio_file}")

    model_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_checkpoint(
        checkpoint_path,
        map_location=model_device,
    )
    model.to(model_device)
    model.eval()

    encoder = EncodecModel.encodec_model_24khz()
    encoder.set_target_bandwidth(encoder_bandwidth)
    encoder = encoder.to(model_device)
    encoder.eval()

    waveform, original_sample_rate = load_audio_segment(
        audio_file_path,
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
        scale_values=scale_values,
        duration_seconds=duration_seconds,
    )
    save_image(image_array, output_image_path)
    return prediction


if __name__ == "__main__":
    # run_inference(
    #     checkpoint="../resources/checkpoints/pitch_detection_supervised/token_transformer_3000.pt",
    #     audio_file="../resources/Gentle on My Mind - The Petersens/Gentle on My Mind - The Petersens.mp3",
    #     output_image="../resources/Gentle on My Mind - The Petersens/pitch_detection_supervised.png",
    #     scale_values=100
    #     # analysis_duration=30,
    # )

    run_inference(
        checkpoint="../resources/checkpoints/pitch_detection_supervised/token_transformer_3000.pt",
        audio_file="../resources/Phases - TWO LANES/Phases - TWO LANES.mp3",
        output_image="../resources/Phases - TWO LANES/pitch_detection_supervised.png",
        scale_values=100
    )
