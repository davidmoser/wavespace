import copy
import math
import os
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .configuration import Config
from .evaluate import (
    evaluate,
    soft_targets,
    _build_space_and_delta,
    _compute_batch_metrics,
)
from .make_loaders import make_loaders, pad_or_crop

def _resolve_device(config: Config) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: Config,
) -> Dict[str, Optional[float]]:
    config.validate()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    device = _resolve_device(config)
    model.to(device)

    if hasattr(model, "seq_len"):
        if model.seq_len != config.seq_len:
            raise ValueError("Model seq_len does not match config.seq_len")
    if hasattr(model, "latent_dim"):
        if model.latent_dim != config.latent_dim:
            raise ValueError("Model latent_dim does not match config.latent_dim")
    if hasattr(model, "n_classes"):
        if model.n_classes != config.n_classes:
            raise ValueError("Model n_classes does not match config.n_classes")

    if not isinstance(train_dataloader, DataLoader):
        raise TypeError("train_dataloader must be an instance of torch.utils.data.DataLoader")

    if val_dataloader is not None and not isinstance(val_dataloader, DataLoader):
        raise TypeError("val_dataloader must be a DataLoader or None")

    train_loader = train_dataloader
    val_loader = val_dataloader

    total_train_steps = len(train_loader) * config.epochs
    if total_train_steps == 0:
        raise ValueError("Training loader must yield at least one batch per epoch")
    total_steps = config.total_steps_override or total_train_steps

    centers_hz = torch.tensor(config.centers_hz(), dtype=torch.float32, device=device)
    space_centers, _ = _build_space_and_delta(centers_hz, config.log_bins)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = device.type == "cuda" and config.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            x = batch["x"].to(device, non_blocking=True)
            freq = batch["freq_hz"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                target = soft_targets(freq, centers_hz, config.sigma_bins, config.log_bins)
                log_q = F.log_softmax(logits, dim=-1)
                frame_loss = -(target * log_q).sum(dim=-1)
                mask_sum = mask.sum()
                if mask_sum <= 0:
                    continue
                loss = (frame_loss * mask).sum() / mask_sum

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            scheduler.step()

            with torch.no_grad():
                top1, within, mae = _compute_batch_metrics(
                    logits, freq, space_centers, config.within_bins, config.log_bins, mask
                )

            if (global_step + 1) % config.log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} Step {global_step + 1}: "
                    f"lr={current_lr:.6f} loss={loss.item():.4f} "
                    f"top1={top1.item():.4f} within={within.item():.4f}"
                )

            should_eval = (global_step + 1) % config.eval_interval == 0
            if should_eval and val_loader is not None:
                val_metrics = evaluate(model, val_loader, config)
                val_loss = val_metrics.get("loss", float("inf"))
                val_top1 = val_metrics.get("top1", 0.0)
                print(
                    f"Validation @ Epoch {epoch} Step {global_step + 1}: "
                    f"loss={val_loss:.4f} top1={val_top1:.4f} "
                    f"within={val_metrics.get('within_k', 0.0):.4f} "
                    f"mae={val_metrics.get('mae_bins', 0.0):.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_top1 = val_top1
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

            global_step += 1

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, config)
            val_loss = val_metrics.get("loss", float("inf"))
            val_top1 = val_metrics.get("top1", 0.0)
            print(
                f"Validation @ Epoch {epoch} End: loss={val_loss:.4f} top1={val_top1:.4f} "
                f"within={val_metrics.get('within_k', 0.0):.4f} mae={val_metrics.get('mae_bins', 0.0):.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_top1 = val_top1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None and config.save:
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    save_path: Optional[str] = None
    if config.save and best_state is not None:
        save_path = config.save_file
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "model_state_dict": best_state,
            "model_class": model.__class__.__name__,
            "centers_hz": config.centers_hz(),
            "config_dict": asdict(config),
        }
        torch.save(payload, save_path)

    return {
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_val_top1": best_val_top1 if best_val_loss != float("inf") else None,
        "save_path": save_path,
    }

