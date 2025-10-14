import copy
import math
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Callable, Tuple, List

import torch
import wandb
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.poly_dataset import PolyphonicAsyncDatasetFromStore
from .configuration import Configuration
from .dilated_tcn import DilatedTCN
from .evaluate import (
    evaluate,
    _compute_batch_metrics,
)
from .local_context_mlp import LocalContextMLP
from .token_transformer import TokenTransformer
from .utils import label_to_tensor

MODEL_REGISTRY = {
    "DilatedTCN": DilatedTCN,
    "LocalContextMLP": LocalContextMLP,
    "TokenTransformer": TokenTransformer,
}


def _create_model(config: Configuration) -> Module:
    return MODEL_REGISTRY[config.model_name](**config.model_config)


def train(config: Configuration) -> Dict[str, Optional[float]]:
    torch.backends.cudnn.benchmark = True

    device = _resolve_device(config)

    centers_hz = config.centers_hz()
    collate_fn = _create_collate_fn(centers_hz, config.sample_duration, config.time_frames, device)
    train_loader = _load_dataset(config.train_dataset_path, config.batch_size, config.num_workers, collate_fn)
    val_loader = None
    if config.val_dataset_path:
        val_loader = _load_dataset(config.val_dataset_path, config.batch_size, config.num_workers, collate_fn)

    model = _create_model(config)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    lr_lambda = _create_lr(config.epochs, len(train_loader), config.warmup_steps, config.total_steps_override)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    current_step = 1
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            latents, targets = batch  # latents: B,L,T, expected: B,F,T
            latents = latents.to(device)
            targets = targets.to(device)
            logits = model(latents)
            loss = criterion(logits, targets)

            loss.backward()
            if config.max_grad_norm is not None and config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                _ = _compute_batch_metrics(logits, targets, centers_hz)  # TODO: metrics to track

            current_lr = optimizer.param_groups[0]["lr"]
            _log_to_wandb(
                {
                    "train/loss": loss.item(),
                    # "train/top1": top1.item(), # TODO: metrics to log
                    "lr": current_lr,
                    "epoch": epoch,
                },
                step=current_step,
            )

            if current_step % config.log_interval == 0:
                print(
                    f"Epoch {epoch} Step {current_step}: "
                    f"lr={current_lr:.6f} loss={loss.item():.4f} "
                )

            should_eval = current_step % config.eval_interval == 0
            if should_eval and val_loader is not None:
                val_metrics = evaluate(model, val_loader, centers_hz)
                val_loss = val_metrics.get("loss", float("inf"))
                print(
                    f"Validation @ Epoch {epoch} Step {current_step}: "
                    f"loss={val_loss:.4f} "
                )
                _log_to_wandb(
                    {
                        "val/loss": val_metrics.get("loss"),
                        "epoch": epoch,
                    },
                    step=current_step + 1,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

            current_step += 1

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, centers_hz)
            val_loss = val_metrics.get("loss", float("inf"))
            print(
                f"Validation @ Epoch {epoch} End: loss={val_loss:.4f} "
            )
            _log_to_wandb(
                {
                    "val/loss": val_metrics.get("loss"),
                    "epoch": epoch,
                },
                step=current_step,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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


def sweep_run():
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection-supervised")
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection-supervised", config=asdict(cfg))
    train(cfg)


def _resolve_device(config: Configuration) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _login_to_wandb() -> None:
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)


def _init_wandb_run(project: str, config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        wandb.init(project=project)
    else:
        wandb.init(project=project, config=config)


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


def _load_dataset(path: str, batch: int, num_workers: int, collate_fn) -> DataLoader:
    dataset = PolyphonicAsyncDatasetFromStore(path)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return loader


def _create_collate_fn(centers_hz: List[float], duration: float, n_frames: int, device: torch.device) -> Callable[
    [List], Tuple]:
    def collate_fn(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y_tensors = [label_to_tensor(label, centers_hz, duration, n_frames, device=device) for label in ys]
        y = torch.stack(y_tensors)
        return x, y

    return collate_fn


def _create_lr(epochs, steps_per_epoch, warmup_steps, total_steps_override=None):
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
