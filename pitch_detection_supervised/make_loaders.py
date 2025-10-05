from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .configuration import Config


def make_loaders(
        train_data: Optional[Iterable[Any]],
        val_data: Optional[Iterable[Any]],
        config: Config,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    collate_fn = _build_collate_fn(config)

    def _resolve_loader(data: Optional[Iterable[Any]], shuffle: bool) -> Optional[DataLoader]:
        if data is None:
            return None
        if isinstance(data, DataLoader):
            return data

        if config.device is None:
            pin_memory = torch.cuda.is_available()
        else:
            try:
                device_type = torch.device(config.device).type
                pin_memory = device_type == "cuda"
            except (RuntimeError, ValueError):
                pin_memory = False

        return DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    return _resolve_loader(train_data, shuffle=True), _resolve_loader(val_data, shuffle=False)

def pad_or_crop(x: Tensor, target_T: int) -> Tensor:
    """Center-crop or pad the time dimension of *x* to *target_T* steps."""

    if x.ndim == 0:
        raise ValueError("pad_or_crop expects at least 1D tensor")

    current_T = x.shape[0]
    if current_T == target_T:
        return x

    if current_T > target_T:
        start = (current_T - target_T) // 2
        end = start + target_T
        return x[start:end]

    pad_total = target_T - current_T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_shape = list(x.shape)
    pad_shape[0] = target_T
    result = x.new_zeros(pad_shape)
    result[pad_left : pad_left + current_T] = x
    return result


def _to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(dtype)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    if isinstance(value, (float, int)):
        return torch.tensor(value, dtype=dtype)
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)!r}")


def _normalize_time_steps(x: Tensor, eps: float = 1e-12) -> Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    norm = norm.clamp_min(eps)
    return x / norm


def _prepare_sample(sample: Any, config: Config) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    seq_len = config.seq_len

    if isinstance(sample, dict):
        x = sample.get("x")
        freq = sample.get("freq_hz")
        valid = sample.get("valid_mask")
    elif isinstance(sample, (list, tuple)):
        if len(sample) < 2:
            raise ValueError("Sample tuple must contain at least (x, freq_hz)")
        x, freq = sample[0], sample[1]
        valid = sample[2] if len(sample) > 2 else None
    else:
        raise TypeError("Unsupported sample type")

    if x is None or freq is None:
        raise ValueError("Sample must provide 'x' and 'freq_hz'")

    x_tensor = _to_tensor(x, dtype=torch.float32)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(-1)
    x_tensor = pad_or_crop(x_tensor, seq_len)
    if x_tensor.shape[-1] != config.latent_dim:
        raise ValueError(
            f"Sample latent dimension {x_tensor.shape[-1]} does not match config.latent_dim={config.latent_dim}"
        )

    freq_tensor = _to_tensor(freq, dtype=torch.float32)
    if freq_tensor.ndim == 0:
        freq_tensor = freq_tensor.repeat(seq_len)
    elif freq_tensor.ndim == 1:
        freq_tensor = pad_or_crop(freq_tensor.unsqueeze(-1), seq_len).squeeze(-1)
    else:
        freq_tensor = pad_or_crop(freq_tensor, seq_len)
        freq_tensor = freq_tensor.squeeze(-1) if freq_tensor.ndim > 1 else freq_tensor
    if freq_tensor.ndim != 1 or freq_tensor.shape[0] != seq_len:
        freq_tensor = freq_tensor.reshape(seq_len)

    if valid is None:
        valid_tensor = torch.ones(seq_len, dtype=torch.float32)
    else:
        valid_tensor = _to_tensor(valid, dtype=torch.float32)
        if valid_tensor.ndim == 0:
            valid_tensor = valid_tensor.repeat(seq_len)
        else:
            valid_tensor = pad_or_crop(valid_tensor, seq_len)
        valid_tensor = valid_tensor.float()
    if valid_tensor.ndim != 1 or valid_tensor.shape[0] != seq_len:
        valid_tensor = valid_tensor.reshape(seq_len)

    return x_tensor, freq_tensor, valid_tensor


def _build_collate_fn(config: Config) -> Callable[[Sequence[Any]], Dict[str, Tensor]]:
    def collate(batch: Sequence[Any]) -> Dict[str, Tensor]:
        xs: List[Tensor] = []
        freqs: List[Tensor] = []
        masks: List[Tensor] = []

        for sample in batch:
            x_tensor, freq_tensor, valid_tensor = _prepare_sample(sample, config)
            xs.append(x_tensor)
            freqs.append(freq_tensor)
            masks.append(valid_tensor)

        x_batch = torch.stack(xs, dim=0)
        freq_batch = torch.stack(freqs, dim=0)
        mask_batch = torch.stack(masks, dim=0).float()

        x_batch = _normalize_time_steps(x_batch)

        return {"x": x_batch, "freq_hz": freq_batch, "valid_mask": mask_batch}

    return collate

