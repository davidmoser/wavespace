import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .configuration import Config

__all__ = [
    "evaluate",
    "soft_targets",
    "_freq_to_space",
    "_build_space_and_delta",
    "_nearest_center_index",
    "_compute_batch_metrics",
]


def _freq_to_space(freq_hz: Tensor, use_log: bool) -> Tensor:
    if use_log:
        return torch.log(freq_hz.clamp_min(1e-12))
    return freq_hz


def _build_space_and_delta(centers_hz_tensor: Tensor, use_log: bool) -> Tuple[Tensor, Tensor]:
    space_centers = _freq_to_space(centers_hz_tensor, use_log)
    if space_centers.numel() <= 1:
        delta = torch.tensor(1.0, dtype=space_centers.dtype, device=space_centers.device)
    else:
        diffs = space_centers[1:] - space_centers[:-1]
        delta = diffs.median()
        if not torch.isfinite(delta):
            delta = torch.tensor(1.0, dtype=space_centers.dtype, device=space_centers.device)
    return space_centers, delta


def _nearest_center_index(space_value: Tensor, space_centers: Tensor) -> Tensor:
    diff = space_value.unsqueeze(-1) - space_centers
    return torch.argmin(diff.abs(), dim=-1)


def soft_targets(freq_hz: Tensor, centers_hz: Tensor, sigma_bins: float, use_log: bool) -> Tensor:
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")

    if freq_hz.ndim == 1:
        freq_hz = freq_hz.unsqueeze(-1)
    freq_hz = freq_hz.to(dtype=centers_hz.dtype, device=centers_hz.device)

    space_centers, delta = _build_space_and_delta(centers_hz, use_log)
    freq_space = _freq_to_space(freq_hz, use_log).unsqueeze(-1)
    d = (freq_space - space_centers) / delta.clamp_min(1e-12)
    weights = torch.exp(-0.5 * (d / sigma_bins) ** 2)
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / weights_sum


def _masked_average(value: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum()
    if denom <= 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return (value * mask).sum() / denom


def _compute_batch_metrics(
    logits: Tensor,
    freq_hz: Tensor,
    space_centers: Tensor,
    within_bins: int,
    use_log: bool,
    mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    pred_idx = logits.argmax(dim=-1)
    freq_space = _freq_to_space(freq_hz, use_log)
    target_idx = _nearest_center_index(freq_space, space_centers)
    diff = (pred_idx - target_idx).abs().float()

    top1 = (diff == 0).float()
    within = (diff <= within_bins).float()

    mask = mask.float()
    top1_mean = _masked_average(top1, mask)
    within_mean = _masked_average(within, mask)
    mae_mean = _masked_average(diff, mask)

    return top1_mean, within_mean, mae_mean


@torch.no_grad()
def evaluate(model: Module, data_loader: Optional[DataLoader], config: Config) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": math.nan, "top1": math.nan, "within_k": math.nan, "mae_bins": math.nan}

    model.eval()
    device = next(model.parameters()).device

    centers_hz = torch.tensor(config.centers_hz(), dtype=torch.float32, device=device)
    space_centers, _ = _build_space_and_delta(centers_hz, config.log_bins)

    total_loss = torch.zeros(1, device=device)
    total_mask = torch.zeros(1, device=device)
    total_top1 = torch.zeros(1, device=device)
    total_within = torch.zeros(1, device=device)
    total_mae = torch.zeros(1, device=device)

    for batch in data_loader:
        x = batch["x"].to(device, non_blocking=True)
        freq = batch["freq_hz"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True).float()

        logits = model(x)
        target = soft_targets(freq, centers_hz, config.sigma_bins, config.log_bins)
        log_q = F.log_softmax(logits, dim=-1)
        frame_loss = -(target * log_q).sum(dim=-1)

        mask_sum = mask.sum()
        total_loss += (frame_loss * mask).sum()
        total_mask += mask_sum

        top1, within, mae = _compute_batch_metrics(
            logits, freq, space_centers, config.within_bins, config.log_bins, mask
        )

        total_top1 += top1 * mask_sum
        total_within += within * mask_sum
        total_mae += mae * mask_sum

    denom = total_mask.clamp_min(1.0)
    metrics = {
        "loss": (total_loss / denom).item(),
        "top1": (total_top1 / denom).item(),
        "within_k": (total_within / denom).item(),
        "mae_bins": (total_mae / denom).item(),
    }
    return metrics
