"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterator

import pytest
import torch

from datasets.create_latent_store import create_latent_store
from datasets.poly_dataset import (
    PolyphonicAsyncDataset,
    PolyphonicAsyncDatasetFromStore,
)


@pytest.fixture()
def dummy_encodec(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install a lightweight stand-in for the ``encodec`` package."""

    class _DummyEncodecModel:
        def __init__(self) -> None:
            self.sample_rate = 16_000
            self.channels = 1
            self.bandwidth = 24.0

        def set_target_bandwidth(self, bandwidth: float) -> None:
            self.bandwidth = bandwidth

        def to(self, device: torch.device | str) -> "_DummyEncodecModel":
            return self

        def eval(self) -> "_DummyEncodecModel":
            return self

        def encoder(self, waveform: torch.Tensor) -> torch.Tensor:
            return waveform

    encodec_module = types.ModuleType("encodec")

    class _EncodecModelFactory:
        @staticmethod
        def encodec_model_24khz() -> _DummyEncodecModel:
            return _DummyEncodecModel()

    encodec_module.EncodecModel = _EncodecModelFactory  # type: ignore[attr-defined]

    utils_module = types.ModuleType("encodec.utils")

    def convert_audio(
        waveform: torch.Tensor,
        *_: object,
        **__: object,
    ) -> torch.Tensor:
        return waveform

    utils_module.convert_audio = convert_audio  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "encodec", encodec_module)
    monkeypatch.setitem(sys.modules, "encodec.utils", utils_module)

    try:
        yield
    finally:
        monkeypatch.delitem(sys.modules, "encodec", raising=False)
        monkeypatch.delitem(sys.modules, "encodec.utils", raising=False)


def test_polyphonic_dataset_store_roundtrip(tmp_path: Path, dummy_encodec: None) -> None:
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
