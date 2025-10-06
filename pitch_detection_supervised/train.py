import copy
import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from .configuration import Config
from .evaluate import (
    evaluate,
    soft_targets,
    _build_space_and_delta,
    _compute_batch_metrics,
)


ModelFactory = Callable[[Config], Module]
LoaderFactory = Callable[[Config], Optional[DataLoader]]
ModelLike = Union[Module, ModelFactory]
LoaderLike = Union[Optional[DataLoader], LoaderFactory]


def _login_to_wandb() -> None:
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)


def _init_wandb_run(project: str, config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        wandb.init(project=project)
    else:
        wandb.init(project=project, config=config)


def _wandb_config_dict() -> Dict[str, Any]:
    cfg = wandb.config
    if hasattr(cfg, "as_dict"):
        return cfg.as_dict()
    return dict(cfg)


def _log_to_wandb(metrics: Dict[str, Any], step: int) -> None:
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


def _update_wandb_summary(summary: Dict[str, Any]) -> None:
    if not wandb.run:
        return
    for key, value in summary.items():
        if value is None:
            continue
        wandb.run.summary[key] = value


def _resolve_model(model_like: ModelLike, config: Config) -> Module:
    if isinstance(model_like, Module):
        return model_like
    if callable(model_like):
        model = model_like(config)
        if not isinstance(model, Module):
            raise TypeError("Model factory must return an instance of torch.nn.Module")
        return model
    raise TypeError("model must be an nn.Module or a callable returning one")


def _resolve_loader(loader_like: LoaderLike, config: Config) -> Optional[DataLoader]:
    if loader_like is None:
        return None
    if isinstance(loader_like, DataLoader):
        return loader_like
    if callable(loader_like):
        loader = loader_like(config)
        if loader is not None and not isinstance(loader, DataLoader):
            raise TypeError("Loader factory must return a torch.utils.data.DataLoader or None")
        return loader
    raise TypeError("loader must be a DataLoader, callable, or None")


def _resolve_device(config: Config) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
        model: Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
) -> Dict[str, Optional[float]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    device = _resolve_device(config)
    model.to(device)

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

            logits = model(x)
            target = soft_targets(freq, centers_hz, config.sigma_bins, config.log_bins)
            log_q = F.log_softmax(logits, dim=-1)
            frame_loss = -(target * log_q).sum(dim=-1)
            mask_sum = mask.sum()
            if mask_sum <= 0:
                continue
            loss = (frame_loss * mask).sum() / mask_sum

            loss.backward()
            if config.max_grad_norm is not None and config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                top1, within, mae = _compute_batch_metrics(
                    logits, freq, space_centers, config.within_bins, config.log_bins, mask
                )

            current_lr = optimizer.param_groups[0]["lr"]
            _log_to_wandb(
                {
                    "train/loss": loss.item(),
                    "train/top1": top1.item(),
                    "train/within_k": within.item(),
                    "train/mae_bins": mae.item(),
                    "train/mask_fraction": float(mask_sum.item() / mask.numel()),
                    "lr": current_lr,
                    "epoch": epoch,
                },
                step=global_step + 1,
            )

            if (global_step + 1) % config.log_interval == 0:
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
                _log_to_wandb(
                    {
                        "val/loss": val_metrics.get("loss"),
                        "val/top1": val_metrics.get("top1"),
                        "val/within_k": val_metrics.get("within_k"),
                        "val/mae_bins": val_metrics.get("mae_bins"),
                        "epoch": epoch,
                    },
                    step=global_step + 1,
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
            _log_to_wandb(
                {
                    "val/loss": val_metrics.get("loss"),
                    "val/top1": val_metrics.get("top1"),
                    "val/within_k": val_metrics.get("within_k"),
                    "val/mae_bins": val_metrics.get("mae_bins"),
                    "epoch": epoch,
                },
                step=global_step,
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

    results = {
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_val_top1": best_val_top1 if best_val_loss != float("inf") else None,
        "save_path": save_path,
    }
    _update_wandb_summary(results)
    return results


def single_run(
        config: Config,
        model: ModelLike,
        train_loader: LoaderLike,
        val_loader: LoaderLike = None,
        project: str = "pitch-detection-supervised",
) -> Dict[str, Optional[float]]:
    """Execute a single Weights & Biases run using the provided resources."""

    _login_to_wandb()
    _init_wandb_run(project=project, config=asdict(config))

    try:
        resolved_model = _resolve_model(model, config)
        resolved_train_loader = _resolve_loader(train_loader, config)
        if resolved_train_loader is None:
            raise ValueError("train_loader must be provided")
        resolved_val_loader = _resolve_loader(val_loader, config)
        results = train(resolved_model, resolved_train_loader, resolved_val_loader, config)
    finally:
        if wandb.run:
            wandb.finish()

    return results


def sweep_run(
        model_factory: ModelFactory,
        train_loader_factory: LoaderFactory,
        val_loader_factory: Optional[LoaderFactory] = None,
        base_config: Optional[Config] = None,
        project: str = "pitch-detection-supervised",
) -> Dict[str, Optional[float]]:
    """Execute a sweep-configured Weights & Biases run."""

    _login_to_wandb()
    init_config = asdict(base_config) if base_config is not None else None
    _init_wandb_run(project=project, config=init_config)

    cfg_dict = _wandb_config_dict()
    if base_config is not None:
        merged = dict(init_config or {})
        merged.update(cfg_dict)
        cfg_dict = merged

    config = Config.from_dict(cfg_dict)

    try:
        resolved_model = _resolve_model(model_factory, config)
        resolved_train_loader = _resolve_loader(train_loader_factory, config)
        if resolved_train_loader is None:
            raise ValueError("train_loader_factory must provide a DataLoader")
        resolved_val_loader = _resolve_loader(val_loader_factory, config)
        results = train(resolved_model, resolved_train_loader, resolved_val_loader, config)
    finally:
        if wandb.run:
            wandb.finish()

    return results
