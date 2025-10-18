import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import numpy as np
import wandb
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from .utils import label_to_tensor, log_to_wandb

LOG_SAMPLE_SEED = 2024
LOG_SAMPLE_COUNT = 5


@dataclass
class _LoggingSample:
    index: int
    latents: Tensor
    target: Tensor


@torch.no_grad()
def evaluate(model: Module, data_loader: Optional[DataLoader], centers_hz: List[float]) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": math.nan}

    device = next(model.parameters()).device
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = torch.zeros(1, device=device)

    count = 0
    for batch in data_loader:
        latents, targets = batch

        logits = model(latents)
        loss = criterion(logits, targets)

        total_loss += loss.sum()
        count += 1

        _ = _compute_batch_metrics(
            logits, targets, centers_hz
        )

    metrics = {
        "loss": (total_loss / count).item(),
    }
    return metrics


def _compute_batch_metrics(
        logits: Tensor,
        target: Tensor,
        space_centers: List[float],
) -> Tuple[Tensor, Tensor, Tensor]:
    pass  # TODO


def _prepare_logging_samples(
        dataset: Optional[Dataset],
        centers_hz: List[float],
        duration: float,
        n_frames: int,
        seed: int = LOG_SAMPLE_SEED,
        count: int = LOG_SAMPLE_COUNT,
) -> List[_LoggingSample]:
    if dataset is None:
        return []

    dataset_length = len(dataset)
    if dataset_length == 0 or count <= 0:
        return []

    n_select = min(dataset_length, count)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(dataset_length, generator=generator)[:n_select].tolist()

    samples: List[_LoggingSample] = []
    for idx in indices:
        latents, labels = dataset[idx]
        latents_tensor = latents.detach().to(dtype=torch.float32).cpu()
        target_tensor = label_to_tensor(labels, centers_hz, duration, n_frames, device=torch.device("cpu"))
        samples.append(_LoggingSample(index=idx, latents=latents_tensor, target=target_tensor))
    return samples


def _log_evaluation_samples(
        model: Module,
        device: torch.device,
        step: int,
        train_samples: List[_LoggingSample],
        val_samples: List[_LoggingSample],
) -> None:
    _log_sample_predictions(model, train_samples, device, step, "train")
    _log_sample_predictions(model, val_samples, device, step, "val")


def _log_sample_predictions(
        model: Module,
        samples: List[_LoggingSample],
        device: torch.device,
        step: int,
        split: str,
) -> None:
    if not samples or not wandb.run:
        return

    previous_mode = model.training
    model.eval()

    images: List[wandb.Image] = []
    with torch.no_grad():
        for sample_idx, sample in enumerate(samples, start=1):
            inputs = sample.latents.unsqueeze(0).to(device)
            logits = model(inputs)
            prediction = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            target = sample.target.cpu().numpy()
            panel = _create_piano_roll_panel(target, prediction)
            caption = (
                f"{split} sample {sample_idx} (idx={sample.index})\n"
                "Top: target, Bottom: prediction"
            )
            images.append(wandb.Image(panel, caption=caption))

    if images:
        log_to_wandb({f"{split}/piano_rolls": images}, step=step)

    if previous_mode:
        model.train()


def _create_piano_roll_panel(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target_img = np.clip(target, 0.0, 1.0).astype(np.float32)
    prediction_img = np.clip(prediction, 0.0, 1.0).astype(np.float32)
    if target_img.shape != prediction_img.shape:
        min_rows = min(target_img.shape[0], prediction_img.shape[0])
        min_cols = min(target_img.shape[1], prediction_img.shape[1])
        target_img = target_img[:min_rows, :min_cols]
        prediction_img = prediction_img[:min_rows, :min_cols]

    separator = np.ones((1, target_img.shape[1]), dtype=np.float32)
    panel = np.concatenate((target_img, separator, prediction_img), axis=0)
    return panel
