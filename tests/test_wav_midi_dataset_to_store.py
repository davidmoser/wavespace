"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""

from pathlib import Path

import torch

from datasets.create_latent_store import create_latent_store
from datasets.latent_salience_store import LatentSalienceStore
from datasets.wav_midi_salience_dataset import WavMidiSalienceDataset


def test_wav_midi_dataset_to_store(tmp_path: Path) -> None:
    dataset = WavMidiSalienceDataset(
        wav_midi_path="./resources/maestro-v3.0.0",
        n_samples=10,
        sample_rate=44_100,
        duration=20,
        label_sample_rate=75,
        label_type="activation",
    )

    output_path = tmp_path / "poly_store"
    output_path.mkdir()

    captured_latents: dict[int, torch.Tensor] = {}

    def capture_latents(index: int, latents: torch.Tensor) -> None:
        captured_latents[index] = latents.clone()

    create_latent_store(
        dataset,
        output_path,
        dataset_sample_rate=dataset.sample_rate,
        samples_per_shard=4,
        latent_callback=capture_latents,
    )

    store = LatentSalienceStore(output_path)

    assert len(store) == len(dataset)

    iterator = iter(dataset)
    for index in range(len(dataset)):
        _, expected_label = next(iterator)
        stored_latents, stored_label = store[index]

        torch.testing.assert_close(stored_latents, captured_latents[index])
        torch.testing.assert_close(stored_label, expected_label)
