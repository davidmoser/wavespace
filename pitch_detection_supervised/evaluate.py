import math
from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: Module, data_loader: Optional[DataLoader], centers_hz: List[float]) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": math.nan}

    model.eval()
    device = next(model.parameters()).device
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = torch.zeros(1, device=device)

    count = 0
    for batch in data_loader:
        latents, targets = batch

        logits = model(latents)
        loss = criterion(logits, targets)

        total_loss += loss.sum()
        count += 1

        _ = _compute_batch_metrics(
            logits, targets, centers_hz
        )

    metrics = {
        "loss": (total_loss / count).item(),
    }
    return metrics


def _compute_batch_metrics(
        logits: Tensor,
        target: Tensor,
        space_centers: List[float],
) -> Tuple[Tensor, Tensor, Tensor]:
    pass  # TODO
