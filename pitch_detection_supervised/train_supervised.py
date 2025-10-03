import copy
import dataclasses
import math
import random
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader


def pad_or_crop(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Center-crop or zero-pad ``x`` along the time dimension to ``target_T``."""
    if x.dim() == 0:
        raise ValueError("pad_or_crop expects at least 1D tensor")
    current_T = x.shape[0]
    if current_T == target_T:
        return x
    start_src, end_src, start_dst, end_dst = _compute_spans(current_T, target_T)
    if current_T >= target_T:
        return x[start_src:end_src]
    out_shape = (target_T, *x.shape[1:])
    out = x.new_zeros(out_shape)
    out[start_dst:end_dst] = x[start_src:end_src]
    return out


def _compute_spans(current_T: int, target_T: int) -> tuple[int, int, int, int]:
    if current_T >= target_T:
        start_src = (current_T - target_T) // 2
        end_src = start_src + target_T
        start_dst = 0
        end_dst = target_T
    else:
        start_src = 0
        end_src = current_T
        pad_total = target_T - current_T
        start_dst = pad_total // 2
        end_dst = start_dst + current_T
    return start_src, end_src, start_dst, end_dst


def _make_loaders(train_ds: Any, val_ds: Any, config: Any) -> tuple[DataLoader, Optional[DataLoader]]:
    batch_size = int(getattr(config, "batch_size", 1))
    num_workers = int(getattr(config, "num_workers", 0))
    seq_len = int(getattr(config, "seq_len"))
    device_str = getattr(config, "device", None)
    if device_str in (None, "auto"):
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = str(device_str).startswith("cuda")

    def _worker_init_fn(worker_id: int) -> None:
        seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    def _as_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.to(torch.float32)
        tensor = tensor.to(dtype)
        return tensor

    def _collate(samples: Sequence[Any]) -> dict[str, torch.Tensor]:
        xs: list[torch.Tensor] = []
        freqs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for sample in samples:
            if isinstance(sample, dict):
                x_val = sample["x"]
                freq_val = sample["freq_hz"]
                mask_val = sample.get("valid_mask")
            else:
                if len(sample) < 2:
                    raise ValueError("Dataset samples must provide (x, freq_hz)")
                x_val, freq_val = sample[:2]
                mask_val = sample[2] if len(sample) > 2 else None

            x_tensor = _as_tensor(x_val, torch.float32)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(-1)
            current_T = x_tensor.shape[0]
            start_src, end_src, start_dst, end_dst = _compute_spans(current_T, seq_len)
            x_tensor = pad_or_crop(x_tensor, seq_len)
            x_tensor = _l2_normalize(x_tensor)
            xs.append(x_tensor)

            base_mask = x_tensor.new_zeros(seq_len)
            base_mask[start_dst:end_dst] = 1.0

            if isinstance(freq_val, (float, int)):
                freq_tensor = torch.full((seq_len,), float(freq_val), dtype=torch.float32)
            else:
                freq_tensor = _as_tensor(freq_val, torch.float32)
                if freq_tensor.dim() == 0:
                    freq_tensor = torch.full((seq_len,), float(freq_tensor.item()), dtype=torch.float32)
                else:
                    if freq_tensor.dim() > 1 and freq_tensor.shape[-1] == 1:
                        freq_tensor = freq_tensor.squeeze(-1)
                    freq_tensor = pad_or_crop(freq_tensor, seq_len)
            freqs.append(freq_tensor)

            if mask_val is None:
                mask_tensor = base_mask
            else:
                mask_tensor = _as_tensor(mask_val, torch.float32)
                if mask_tensor.dim() == 0:
                    mask_tensor = torch.full((seq_len,), float(mask_tensor.item()), dtype=torch.float32)
                else:
                    if mask_tensor.dim() > 1 and mask_tensor.shape[-1] == 1:
                        mask_tensor = mask_tensor.squeeze(-1)
                    mask_tensor = pad_or_crop(mask_tensor, seq_len)
                mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0) * base_mask
            masks.append(mask_tensor)

        batch_x = torch.stack(xs)
        batch_freq = torch.stack(freqs)
        batch_mask = torch.stack(masks)
        return {"x": batch_x, "freq_hz": batch_freq, "valid_mask": batch_mask}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    val_loader: Optional[DataLoader]
    if val_ds is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            collate_fn=_collate,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    return train_loader, val_loader


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm


def _build_space_and_delta(
    centers_hz_tensor: torch.Tensor, use_log: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers_hz_tensor.numel() == 0:
        raise ValueError("centers_hz_tensor must contain at least one value")
    if use_log:
        space_centers = torch.log(torch.clamp(centers_hz_tensor, min=1e-12))
    else:
        space_centers = centers_hz_tensor
    if space_centers.numel() < 2:
        delta = torch.tensor(1.0, dtype=space_centers.dtype, device=space_centers.device)
    else:
        diffs = space_centers.diff()
        delta = torch.median(diffs)
    delta = torch.clamp(delta, min=1e-6)
    return space_centers, delta


def _nearest_center_index(space_value: torch.Tensor, space_centers: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(space_value.unsqueeze(-1) - space_centers)
    return torch.argmin(diff, dim=-1)


def soft_targets(
    freq_hz: torch.Tensor,
    centers_hz: torch.Tensor,
    sigma_bins: float,
    use_log: bool,
) -> torch.Tensor:
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    freq_tensor = torch.as_tensor(freq_hz, dtype=torch.float32, device=centers_hz.device)
    if freq_tensor.dim() == 1:
        freq_tensor = freq_tensor.unsqueeze(-1)
    freq_tensor = torch.clamp(freq_tensor, min=1e-8)

    space_centers, delta = _build_space_and_delta(centers_hz, use_log)
    if use_log:
        space_freq = torch.log(freq_tensor)
    else:
        space_freq = freq_tensor
    diff_bins = (space_freq.unsqueeze(-1) - space_centers) / delta
    weights = torch.exp(-0.5 * (diff_bins / float(sigma_bins)) ** 2)
    weights_sum = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
    return weights / weights_sum


def evaluate(model: nn.Module, data_loader: Optional[DataLoader], config: Any) -> dict[str, float]:
    if data_loader is None:
        raise ValueError("data_loader must not be None for evaluation")
    device = next(model.parameters()).device
    model.eval()

    centers_hz_tensor = _centers_hz_tensor(config, device)
    use_log = bool(getattr(config, "log_bins", False))
    sigma_bins = float(getattr(config, "sigma_bins"))
    space_centers, _ = _build_space_and_delta(centers_hz_tensor, use_log)

    total_loss = 0.0
    total_frames = 0.0
    total_top1 = 0.0
    total_within = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            freq = batch["freq_hz"].to(device)
            mask = batch["valid_mask"].to(device)
            logits = model(x)
            targets = soft_targets(freq, centers_hz_tensor, sigma_bins, use_log)
            log_q = F.log_softmax(logits, dim=-1)
            loss = -(targets * log_q).sum(dim=-1)
            loss = loss * mask

            frame_count = mask.sum().item()
            if frame_count <= 0:
                continue

            total_loss += loss.sum().item()
            total_frames += frame_count

            space_freq = torch.log(torch.clamp(freq, min=1e-8)) if use_log else freq
            target_idx = _nearest_center_index(space_freq, space_centers)
            pred_idx = torch.argmax(logits, dim=-1)
            diff = torch.abs(pred_idx - target_idx).float()

            total_top1 += (mask * (diff == 0).float()).sum().item()
            within_bins = int(getattr(config, "within_bins", 0))
            total_within += (mask * (diff <= within_bins).float()).sum().item()
            total_mae += (mask * diff).sum().item()

    avg_loss = total_loss / total_frames if total_frames > 0 else 0.0
    avg_top1 = total_top1 / total_frames if total_frames > 0 else 0.0
    avg_within = total_within / total_frames if total_frames > 0 else 0.0
    avg_mae = total_mae / total_frames if total_frames > 0 else 0.0
    return {
        "loss": avg_loss,
        "top1": avg_top1,
        "within_k": avg_within,
        "mae_bins": avg_mae,
    }


def train(
    model: nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    config: Any,
) -> dict[str, Any]:
    seed = int(getattr(config, "seed", 0))
    _set_global_seed(seed)
    torch.backends.cudnn.benchmark = True

    device = _resolve_device(getattr(config, "device", None))
    model.to(device)

    seq_len = int(getattr(config, "seq_len"))
    latent_dim = getattr(config, "latent_dim", None)
    n_classes = getattr(config, "n_classes", None)

    if hasattr(model, "seq_len"):
        assert int(getattr(model, "seq_len")) == seq_len, "Model seq_len mismatch"
    if latent_dim is not None and hasattr(model, "latent_dim"):
        assert int(getattr(model, "latent_dim")) == int(latent_dim), "Model latent_dim mismatch"
    if n_classes is not None and hasattr(model, "n_classes"):
        assert int(getattr(model, "n_classes")) == int(n_classes), "Model n_classes mismatch"

    train_loader, val_loader = _make_loaders(train_dataset, val_dataset, config)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training dataset must yield at least one batch")
    epochs = int(getattr(config, "epochs"))
    warmup_steps = int(getattr(config, "warmup_steps", 0))
    total_steps_override = getattr(config, "total_steps_override", None)
    total_steps = int(total_steps_override) if total_steps_override else epochs * steps_per_epoch

    lr = float(getattr(config, "lr"))
    weight_decay = float(getattr(config, "weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = GradScaler(enabled=bool(getattr(config, "use_amp", False)) and device.type == "cuda")

    centers_hz_tensor = _centers_hz_tensor(config, device)
    use_log = bool(getattr(config, "log_bins", False))
    sigma_bins = float(getattr(config, "sigma_bins"))
    space_centers, _ = _build_space_and_delta(centers_hz_tensor, use_log)
    within_bins = int(getattr(config, "within_bins", 0))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    last_val_metrics = {"loss": float("inf"), "top1": 0.0, "within_k": 0.0, "mae_bins": 0.0}

    log_interval = int(getattr(config, "log_interval", 100))
    eval_interval = int(getattr(config, "eval_interval", 0))

    def _compute_lr(step: int) -> float:
        if total_steps <= 0:
            return lr
        if step < warmup_steps and warmup_steps > 0:
            return lr * float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    step = 0
    meter = {"loss": 0.0, "top1": 0.0, "within": 0.0, "frames": 0.0}
    last_train_loss = 0.0
    last_train_top1 = 0.0
    last_train_within_k = 0.0
    last_train_mae = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            current_lr = _compute_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            x = batch["x"].to(device)
            freq = batch["freq_hz"].to(device)
            mask = batch["valid_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                targets = soft_targets(freq, centers_hz_tensor, sigma_bins, use_log)
                log_q = F.log_softmax(logits, dim=-1)
                loss = -(targets * log_q).sum(dim=-1)
                loss = (loss * mask)
                frame_count = mask.sum()
                if frame_count > 0:
                    loss = loss.sum() / frame_count
                else:
                    loss = loss.mean()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                logits_detached = logits.detach()
                space_freq = torch.log(torch.clamp(freq, min=1e-8)) if use_log else freq
                target_idx = _nearest_center_index(space_freq, space_centers)
                pred_idx = torch.argmax(logits_detached, dim=-1)
                diff = torch.abs(pred_idx - target_idx).float()
                valid = mask.float()
                frames = valid.sum().item()
                top1_count = float((valid * (diff == 0).float()).sum().item())
                within_count = float((valid * (diff <= within_bins).float()).sum().item())
                mae_total = float((valid * diff).sum().item())
                meter["loss"] += float(loss.detach().item()) * frames
                meter["top1"] += top1_count
                meter["within"] += within_count
                meter["frames"] += frames
                if frames > 0:
                    last_train_loss = float(loss.detach().item())
                    last_train_top1 = top1_count / frames
                    last_train_within_k = within_count / frames
                    last_train_mae = mae_total / frames

            step += 1

            if log_interval > 0 and step % log_interval == 0:
                avg_loss = meter["loss"] / meter["frames"] if meter["frames"] > 0 else 0.0
                avg_top1 = meter["top1"] / meter["frames"] if meter["frames"] > 0 else 0.0
                avg_within = meter["within"] / meter["frames"] if meter["frames"] > 0 else 0.0
                print(
                    f"Epoch {epoch} Step {step}/{total_steps}: lr={current_lr:.6f} "
                    f"loss={avg_loss:.4f} top1={avg_top1:.4f} within={avg_within:.4f}"
                )
                meter = {"loss": 0.0, "top1": 0.0, "within": 0.0, "frames": 0.0}

            if eval_interval > 0 and step % eval_interval == 0 and val_loader is not None:
                val_metrics = evaluate(model, val_loader, config)
                last_val_metrics = val_metrics
                print(
                    f"Validation @ step {step}: loss={val_metrics['loss']:.4f} "
                    f"top1={val_metrics['top1']:.4f} within={val_metrics['within_k']:.4f} "
                    f"mae={val_metrics['mae_bins']:.4f}"
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_val_top1 = val_metrics["top1"]
                    best_state = copy.deepcopy(model.state_dict())

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, config)
            last_val_metrics = val_metrics
            print(
                f"Validation @ epoch {epoch}: loss={val_metrics['loss']:.4f} "
                f"top1={val_metrics['top1']:.4f} within={val_metrics['within_k']:.4f} "
                f"mae={val_metrics['mae_bins']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_top1 = val_metrics["top1"]
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    save_path: Optional[str] = None
    if bool(getattr(config, "save", False)):
        save_file = getattr(config, "save_file", None)
        if save_file is None:
            raise ValueError("config.save_file must be provided when save is True")
        save_path = str(save_file)
        _ensure_parent_dir(save_path)
        payload = {
            "model_state_dict": best_state,
            "model_class": model.__class__.__qualname__,
            "centers_hz": list(map(float, centers_hz_tensor.tolist())),
            "config_dict": _config_to_dict(config),
        }
        torch.save(payload, save_path)

    summary = {
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "last_val_loss": last_val_metrics["loss"],
        "last_val_top1": last_val_metrics["top1"],
        "last_val_within_k": last_val_metrics["within_k"],
        "last_val_mae_bins": last_val_metrics["mae_bins"],
        "last_train_loss": last_train_loss,
        "last_train_top1": last_train_top1,
        "last_train_within_k": last_train_within_k,
        "last_train_mae_bins": last_train_mae,
        "save_path": save_path,
    }
    return summary


def _centers_hz_tensor(config: Any, device: torch.device) -> torch.Tensor:
    centers_attr = getattr(config, "centers_hz")
    centers = centers_attr() if callable(centers_attr) else centers_attr
    return torch.as_tensor(centers, dtype=torch.float32, device=device)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "as_dict") and callable(getattr(config, "as_dict")):
        return dict(config.as_dict())
    if hasattr(config, "__dict__"):
        return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    raise TypeError("Cannot convert config to dict")


def _ensure_parent_dir(path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_pref: Any) -> torch.device:
    if device_pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_pref, torch.device):
        return device_pref
    return torch.device(device_pref)
