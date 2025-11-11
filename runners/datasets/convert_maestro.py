"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""
import csv
import time
from pathlib import Path

import numpy as np

from datasets.create_latent_store import create_latent_store
from datasets.wav_midi_salience_dataset import WavMidiSalienceDataset


def convert_maestro(
        output_dir: str,
        n_samples: int = 10,
        duration: float = 20,
        label_type: str = "power",
        samples_per_shard: int = 1024,
        encode_batch_size: int = 8,
        num_workers: int = 1,
) -> None:
    dataset = WavMidiSalienceDataset(
        wav_midi_path="../../resources/maestro-v3.0.0",
        n_samples=n_samples,
        sample_rate=44_100,
        duration=duration,
        label_sample_rate=75,
        label_type=label_type,
    )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=False)

    create_latent_store(
        dataset,
        output_path,
        dataset_sample_rate=dataset.sample_rate,
        samples_per_shard=samples_per_shard,
        encode_batch_size=encode_batch_size,
        num_workers=num_workers,
    )


def optimize():
    encode_batch_sizes = [1]
    num_workerss = [1]
    times = np.zeros((5, 4), dtype=float)
    for i, encode_batch_size in enumerate(encode_batch_sizes):
        for j, num_workers in enumerate(num_workerss):
            print(f"batch size: {encode_batch_size}, number of workers: {num_workers}")
            start = time.time()
            convert_maestro(
                output_dir=f"../../resources/encodec_latents/maestro_20samples_20seconds_{num_workers}workers_{encode_batch_size}batch",
                n_samples=2,
                duration=20,
                label_type="power",
                encode_batch_size=encode_batch_size,
                num_workers=num_workers,
            )
            end = time.time()
            times[i, j] = end - start
            print(f"Time: {end - start}s")

    with open("../../results/maestro/encoding_timings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "num_workers", "seconds"])
        for i, ebs in enumerate(encode_batch_sizes):
            for j, nw in enumerate(num_workerss):
                w.writerow([ebs, nw, f"{times[i, j]:.6f}"])
    print("\nWrote timings.csv")


if __name__ == "__main__":
    optimize()
