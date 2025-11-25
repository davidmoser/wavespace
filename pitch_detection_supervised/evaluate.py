from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import matplotlib.cm as cm
import numpy as np
import torch
import wandb
from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset

from utils.wandb_basic import log_to_wandb
from .utils import normalize_samples, prepare_labels, scale_2d

LOG_SAMPLE_SEED = 2024
LOG_SAMPLE_COUNT = 5


@dataclass
class _LoggingSample:
    sample: Tensor
    label: Tensor


@torch.no_grad()
def evaluate(model: Module, data_loader: DataLoader, label_max_value: float, bce_pos_weight: float) -> Dict[str, float]:
    device = next(model.parameters()).device
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(bce_pos_weight))

    total_loss = torch.zeros(1, device=device)

    count = 0
    for batch in data_loader:
        samples, labels = batch

        samples = normalize_samples(samples.to(device))
        labels = prepare_labels(labels.to(device), samples, label_max_value)

        logits = model(samples)
        loss = criterion(logits, labels)

        total_loss += loss.sum()
        count += 1

        _ = _compute_batch_metrics(logits, labels)

    metrics = {
        "loss": (total_loss / count).item(),
    }
    return metrics


def _compute_batch_metrics(
        logits: Tensor,
        target: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    pass  # TODO


def _prepare_logging_samples(
        dataset: Optional[Dataset],
        label_max_value: float,
        seed: int = LOG_SAMPLE_SEED,
        count: int = LOG_SAMPLE_COUNT,
) -> List[_LoggingSample]:
    loader = DataLoader(dataset, batch_size=count, shuffle=True, generator=torch.Generator().manual_seed(seed))
    samples, labels = next(iter(loader))
    samples = normalize_samples(samples.detach().to(dtype=torch.float32).cpu())
    labels = prepare_labels(labels.detach().to(dtype=torch.float32).cpu(), samples, label_max_value)

    logging_samples: List[_LoggingSample] = []
    for i in range(count):
        logging_samples.append(_LoggingSample(sample=samples[i], label=labels[i]))
    return logging_samples


def _log_evaluation_samples(
        model: Module,
        device: torch.device,
        step: int,
        train_samples: List[_LoggingSample],
        val_samples: List[_LoggingSample],
        incl_sample: bool = False,
) -> None:
    _log_sample_predictions(model, train_samples, device, step, "train", incl_sample)
    _log_sample_predictions(model, val_samples, device, step, "val", incl_sample)


def _log_sample_predictions(
        model: Module,
        logging_samples: List[_LoggingSample],
        device: torch.device,
        step: int,
        split: str,
        incl_sample: bool = False,
) -> None:
    if not logging_samples or not wandb.run:
        return

    previous_mode = model.training
    model.eval()

    images: List[wandb.Image] = []
    with torch.no_grad():
        for sample_idx, logging_sample in enumerate(logging_samples, start=1):
            inputs = logging_sample.sample.unsqueeze(0).to(device)
            logits = model(inputs)
            prediction = torch.sigmoid(logits).squeeze(0).cpu()
            label = logging_sample.label.cpu()
            sample = logging_sample.sample.cpu()
            panel = _create_piano_roll_panel(label, prediction, sample, incl_sample)
            caption = (
                f"{split} sample {sample_idx}\n"
                "Top: label, Middle: prediction, Bottom: sample"
            )
            images.append(wandb.Image(panel, caption=caption))

    if images:
        log_to_wandb({f"{split}/piano_rolls": images}, step=step)

    if previous_mode:
        model.train()


def _create_piano_roll_panel(label: Tensor, prediction: Tensor, sample: Tensor,
                             incl_sample: bool) -> np.ndarray:
    label_img = np.flip(np.clip(label.numpy(), 0.0, 1.0).astype(np.float32), axis=0)
    prediction_img = np.flip(np.clip(prediction.numpy(), 0.0, 1.0).astype(np.float32), axis=0)
    sample = scale_2d(sample, label.shape)
    sample_img = np.flip(np.clip(sample.numpy(), 0.0, 1.0).astype(np.float32), axis=0)

    separator = np.ones((1, label_img.shape[1]), dtype=np.float32)
    panel = np.concatenate((label_img, separator, prediction_img), axis=0)
    if incl_sample:
        panel = np.concatenate((panel, separator, sample_img), axis=0)
    panel_norm = panel / label.max()
    rgba = cm.get_cmap('viridis')(panel_norm)  # (H, W, 4) floats in [0,1]
    rgb = (rgba[..., :3] * 255).astype('uint8')  # (H, W, 3) uint8
    return rgb
