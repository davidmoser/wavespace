import math
import os
from typing import Optional, Dict, Any

import torch
import wandb
from torch import Tensor


def create_warmup_cosine_lr(steps_per_epoch, warmup_steps, epochs=None, steps=None):
    if epochs and steps:
        raise ValueError("epochs and steps cannot both be specified")

    if epochs:
        eff_steps = steps_per_epoch * epochs
        eff_epochs = epochs
    else:
        eff_steps = steps
        eff_epochs = math.ceil(steps / steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, eff_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return eff_epochs, lr_lambda


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
