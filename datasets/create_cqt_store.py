"""Utilities for exporting dataset CQTs alongside MIDI salience labels."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.cqt_midi_salience_dataset import CqtMidiSalienceDataset
from datasets.store_utils import (
    DEFAULT_SAMPLES_PER_SHARD,
    SampleStoreWriter,
    write_global_metadata,
    write_index,
    write_shard_list,
)


def create_cqt_store(
        dataset: CqtMidiSalienceDataset,
        dataset_path: Union[str, Path],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD,
        batch_size: int = 8,
        num_workers: int = 1,
        sample_callback: Optional[Callable[[int, Tensor], None]] = None,
) -> None:
    """Persist a dataset of CQTs and salience tensors as a WebDataset."""

    total_samples = len(dataset)
    dataset_frame_rate = dataset.get_frame_rate()

    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)

    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    shard_count = math.ceil(total_samples / samples_per_shard)
    shard_pad = max(3, len(str(shard_count - 1)))
    key_pad = max(4, len(str(total_samples - 1)))

    store_metadata: Dict[str, Any] = {
        "length": total_samples,
        "dataset_frame_rate": int(dataset_frame_rate),
        "dataset_type": type(dataset).__qualname__,
        "shard_size": int(samples_per_shard),
        "num_shards": shard_count,
        "key_width": key_pad,
        "shard_name_width": shard_pad,
        "created_unix": int(time.time()),
        "version": "1.0",
    }

    combined_metadata: Dict[str, Any] = {"cqt_store": store_metadata}
    if metadata is not None:
        combined_metadata["external"] = dict(metadata)

    workers = 0 if num_workers is None else num_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    writer = SampleStoreWriter(
        path,
        samples_per_shard,
        shard_pad,
        key_pad,
        shard_count,
        sample_key="cqts",
    )

    next_dataset_index = 0

    with torch.inference_mode():
        for batch_samples, batch_labels in loader:
            if isinstance(batch_samples, Tensor):
                batch_len = int(batch_samples.shape[0]) if batch_samples.dim() > 0 else 1
            else:
                batch_len = len(batch_samples)

            if batch_len == 0:
                continue

            print(f"Sample index {next_dataset_index}/{total_samples}")

            writer.write_batch(
                start_index=next_dataset_index,
                samples=batch_samples,
                labels=batch_labels,
                sample_callback=sample_callback,
            )

            next_dataset_index += batch_len

    if next_dataset_index != total_samples:
        raise RuntimeError("Encoded sample count mismatch.")

    shard_paths, global_index = writer.finalize()

    write_global_metadata(path, combined_metadata)
    write_index(path, global_index)
    write_shard_list(path, shard_paths)
