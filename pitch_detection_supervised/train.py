import copy
import math
import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import wandb
from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss
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
from .utils import create_warmup_cosine_lr, log_to_wandb, update_wandb_summary, resolve_device, \
    login_to_wandb

PROJECT_NAME = "pitch-detection-supervised"

MODEL_REGISTRY = {
    "DilatedTCN": DilatedTCN,
    "LocalContextMLP": LocalContextMLP,
    "TokenTransformer": TokenTransformer,
}


def create_model(name: str, config: dict) -> Module:
    return MODEL_REGISTRY[name](**config)


def train(config: Configuration) -> Dict[str, Optional[float]]:
    torch.backends.cudnn.benchmark = True

    device = resolve_device(config.device)

    centers_hz = config.centers_hz()
    train_dataset, val_dataset = _load_datasets(config)
    train_loader = _create_loader(train_dataset, config.batch_size, config.num_workers)
    val_loader = _create_loader(val_dataset, config.batch_size, config.num_workers)

    model = create_model(config.model_name, config.model_config)
    model.to(device)
    model.train()
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(30))
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    epochs, steps, lr_lambda = create_warmup_cosine_lr(len(train_loader), config.warmup_steps, config.epochs,
                                                       config.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_log_samples = _prepare_logging_samples(train_dataset)
    val_log_samples = _prepare_logging_samples(val_dataset)

    current_step = 1
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
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

            if current_step == steps:
                break
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
) -> DataLoader | None:
    if dataset is None:
        return None
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers)
