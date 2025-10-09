"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from datasets.create_latent_store import create_latent_store
from datasets.poly_dataset import (
    PolyphonicAsyncDataset,
    PolyphonicAsyncDatasetFromStore,
)


def test_polyphonic_dataset_store_roundtrip(tmp_path: Path) -> None:
    dataset = PolyphonicAsyncDataset(
        n_samples=10,
        freq_range=(110.0, 220.0),
        max_polyphony=3,
        sr=16_000,
        duration=0.25,
        seed=123,
    )

    output_path = tmp_path / "poly_store"
    output_path.mkdir()

    create_latent_store(
        dataset,
        output_path,
        dataset_sample_rate=dataset.sample_rate,
        samples_per_shard=4,
    )

    store = PolyphonicAsyncDatasetFromStore(output_path)

    assert len(store) == len(dataset)

    for index in range(len(dataset)):
        expected_audio, expected_labels = dataset[index]
        stored_latents, stored_labels = store[index]

        torch.testing.assert_close(stored_latents, expected_audio)
        assert stored_labels == expected_labels
