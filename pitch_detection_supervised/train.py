import copy
import math
import os
from dataclasses import asdict
from typing import Dict, Optional, Callable, Tuple, List

import torch
import wandb
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from datasets.poly_dataset import PolyphonicAsyncDatasetFromStore
from .configuration import Configuration
from .dilated_tcn import DilatedTCN
from .evaluate import (
    evaluate,
    _compute_batch_metrics,
    _prepare_logging_samples,
    _log_evaluation_samples,
)
from .local_context_mlp import LocalContextMLP
from .token_transformer import TokenTransformer
from .utils import label_to_tensor, create_warmup_cosine_lr, log_to_wandb, update_wandb_summary, resolve_device, \
    login_to_wandb

PROJECT_NAME = "pitch-detection-supervised"

MODEL_REGISTRY = {
    "DilatedTCN": DilatedTCN,
    "LocalContextMLP": LocalContextMLP,
    "TokenTransformer": TokenTransformer,
}


def _create_model(config: Configuration) -> Module:
    return MODEL_REGISTRY[config.model_name](**config.model_config)


def train(config: Configuration) -> Dict[str, Optional[float]]:
    torch.backends.cudnn.benchmark = True

    device = resolve_device(config.device)

    centers_hz = config.centers_hz()
    collate_fn = _create_collate_fn(centers_hz, config.sample_duration, config.time_frames, device)
    train_dataset, val_dataset = _load_datasets(config)
    train_loader = _create_loader(train_dataset, config.batch_size, config.num_workers, collate_fn)
    val_loader = _create_loader(val_dataset, config.batch_size, config.num_workers, collate_fn)

    model = _create_model(config)
    model.to(device)
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    lr_lambda = create_warmup_cosine_lr(config.epochs, len(train_loader), config.warmup_steps,
                                        config.total_steps_override)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_log_samples = _prepare_logging_samples(
        train_dataset,
        centers_hz,
        config.sample_duration,
        config.time_frames,
    )
    val_log_samples = _prepare_logging_samples(
        val_dataset,
        centers_hz,
        config.sample_duration,
        config.time_frames,
    )

    current_step = 1
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, config.epochs + 1):
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            latents, targets = batch  # latents: B,L,T, targets: B,F,T
            latents = latents.to(device)
            targets = targets.to(device)
            logits = model(latents)
            loss = criterion(logits, targets)

            loss.backward()
            if config.max_grad_norm is not None and config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                _ = _compute_batch_metrics(logits, targets, centers_hz)  # TODO: metrics to track

            current_lr = optimizer.param_groups[0]["lr"]
            log_to_wandb(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else float(grad_norm),
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
                model.eval()
                val_metrics = evaluate(model, val_loader, centers_hz)
                val_loss = val_metrics.get("loss", float("inf"))
                print(
                    f"Validation @ Epoch {epoch} Step {current_step}: "
                    f"loss={val_loss:.4f} "
                )
                log_to_wandb(
                    {
                        "val/loss": val_metrics.get("loss"),
                        "epoch": epoch,
                    },
                    step=current_step,
                )
                _log_evaluation_samples(
                    model,
                    device,
                    current_step,
                    train_log_samples,
                    val_log_samples,
                )
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

            current_step += 1

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
    update_wandb_summary(results)
    return results


def sweep_run():
    login_to_wandb()
    wandb.init(project=PROJECT_NAME)
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    login_to_wandb()
    wandb.init(project=PROJECT_NAME, config=asdict(cfg))
    train(cfg)


def _load_datasets(config: Configuration) -> Tuple[Dataset, Optional[Dataset]]:
    if config.split_train_set is not None and config.val_dataset_path:
        raise ValueError("Cannot use a validation dataset path and split the training dataset simultaneously.")

    train_dataset = _load_dataset(config.train_dataset_path)
    if config.split_train_set:
        val_length = math.ceil(len(train_dataset) * config.split_train_set)
        train_length = len(train_dataset) - val_length
        return random_split(train_dataset, [train_length, val_length])
    elif config.val_dataset_path:
        val_dataset = _load_dataset(config.val_dataset_path)
        return train_dataset, val_dataset
    else:
        return train_dataset, None


def _load_dataset(path: str) -> Dataset:
    return PolyphonicAsyncDatasetFromStore(path)


def _create_loader(
        dataset: Optional[Dataset],
        batch: int,
        num_workers: int,
        collate_fn,
) -> DataLoader | None:
    if dataset is None:
        return None
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


def _create_collate_fn(centers_hz: List[float], duration: float, n_frames: int, device: torch.device) -> Callable[
    [List], Tuple]:
    def collate_fn(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y_tensors = [label_to_tensor(label, centers_hz, duration, n_frames, device=device) for label in ys]
        y = torch.stack(y_tensors)
        return x, y

    return collate_fn
