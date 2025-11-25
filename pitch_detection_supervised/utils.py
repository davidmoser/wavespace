import math
import os
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Size
from torch import Tensor


def create_warmup_cosine_lr(steps_per_epoch, warmup_fraction, epochs=None, steps=None):
    if epochs and steps:
        raise ValueError("epochs and steps cannot both be specified")

    if epochs:
        eff_steps = steps_per_epoch * epochs
        eff_epochs = epochs
    else:
        eff_steps = steps
        eff_epochs = math.ceil(steps / steps_per_epoch)

    warmup_steps = int(eff_steps * warmup_fraction)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, eff_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return eff_epochs, eff_steps, lr_lambda


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def events_to_active_label(
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


def scale_2d(tensor: Tensor, shape: Size) -> Tensor:
    return scale_3d(tensor.unsqueeze(0), shape).squeeze(0)


def scale_3d(tensor: Tensor, shape: Size) -> Tensor:
    return (F.interpolate(tensor.unsqueeze(0), shape, mode="bilinear", antialias=True, align_corners=False)
            .squeeze(0))


def normalize_samples(samples: torch.Tensor) -> torch.Tensor:
    # samples = torch.log(samples)
    mins = samples.min(dim=1, keepdim=True).values
    maxs = samples.max(dim=1, keepdim=True).values
    samples = (samples - mins) / (maxs - mins + 1e-4)
    return samples


def prepare_labels(labels: torch.Tensor, samples: torch.Tensor, label_max_value: float) -> torch.Tensor:
    scaled_samples = scale_3d(samples, labels.shape[1:3])
    labels = labels * scaled_samples
    return torch.clip(labels / label_max_value, 0, 1)
