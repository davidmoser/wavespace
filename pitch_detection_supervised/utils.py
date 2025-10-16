import math
import os
from typing import Optional, Dict, Any

import torch
import wandb
from torch import Tensor


def create_warmup_cosine_lr(epochs, steps_per_epoch, warmup_steps, total_steps_override=None):
    total_train_steps = steps_per_epoch * epochs
    total_steps = total_steps_override or total_train_steps

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def label_to_tensor(
        events,  # list of (frequency_hz, onset_s, offset_s)
        frequency_bin_centers_hz,  # list/1D array-like
        duration_s: float,
        n_frames: int,
        device: torch.device = None,
        dtype=torch.float32,
):
    # Prepare axes
    f_centers = torch.as_tensor(frequency_bin_centers_hz, dtype=dtype, device=device)
    assert f_centers.ndim == 1, "frequency_bin_centers_hz must be 1D"
    n_f = f_centers.numel()
    y = torch.zeros(n_f, n_frames, dtype=dtype, device=device)
    frame_hop_s = duration_s / n_frames

    # Helper: interpolate a scalar frequency onto neighboring freq bins (handles non-uniform centers)
    def freq_interp_weights(f):
        if f <= f_centers[0]:
            return [(0, 1.0)]
        if f >= f_centers[-1]:
            return [(n_f - 1, 1.0)]
        # searchsorted for right neighbor
        r = int(torch.searchsorted(f_centers, torch.tensor(f, dtype=dtype, device=device)).item())
        l = r - 1
        denom = (f_centers[r] - f_centers[l]).clamp_min(torch.finfo(dtype).eps)
        alpha = float((f - float(f_centers[l])) / float(denom))
        return [(l, 1.0 - alpha), (r, alpha)]

    # Rasterize each event with bilinear (freq × time) interpolation
    for f_hz, t_on, t_off in events:
        # Clamp times to [0, duration]
        t0 = max(0.0, min(float(t_on), duration_s))
        t1 = max(0.0, min(float(t_off), duration_s))
        if not (t1 > t0):
            continue

        # Frequency weights (up to two neighbors)
        f_w = freq_interp_weights(float(f_hz))

        # Time coverage (fractional coverage per frame)
        s = t0 / frame_hop_s
        e = t1 / frame_hop_s
        i_start = max(0, int(math.floor(s)))
        i_end = min(n_frames - 1, int(math.ceil(e)) - 1)
        if i_end < i_start:
            # Entire event falls within a single fractional frame index region
            i_start = i_end = min(n_frames - 1, max(0, int(math.floor((s + e) * 0.5))))

        # Compute overlap per frame (vectorized over the [i_start, i_end] window)
        idx = torch.arange(i_start, i_end + 1, device=device, dtype=dtype)
        # Frame index interval is [i, i+1) in index space
        left = torch.clamp(torch.minimum(torch.maximum(torch.tensor(s, device=device, dtype=dtype), idx), idx + 1.0),
                           min=0.0)
        right = torch.clamp(torch.minimum(torch.maximum(torch.tensor(e, device=device, dtype=dtype), idx), idx + 1.0),
                            min=0.0)
        overlap = (right - left).clamp(min=0.0, max=1.0)  # fractional time weight per frame

        if overlap.numel() == 0:
            continue

        # Add bilinear contributions
        for fi, fw in f_w:
            y[fi, i_start: i_end + 1] += fw * overlap

    # Optionally cap at 1.0 if desired for overlapping events:
    # y.clamp_(0.0, 1.0)

    return y


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def login_to_wandb() -> None:
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)


def init_wandb_run(project: str, config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        wandb.init(project=project)
    else:
        wandb.init(project=project, config=config)


def log_to_wandb(metrics: Dict[str, Any], step: int) -> None:
    if not wandb.run:
        return
    payload = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, Tensor):
            payload[key] = value.detach().item()
        elif isinstance(value, (float, int)):
            payload[key] = float(value)
        else:
            payload[key] = value
    if payload:
        wandb.log(payload, step=step)


def update_wandb_summary(summary: Dict[str, Any]) -> None:
    if not wandb.run:
        return
    for key, value in summary.items():
        if value is None:
            continue
        wandb.run.summary[key] = value
