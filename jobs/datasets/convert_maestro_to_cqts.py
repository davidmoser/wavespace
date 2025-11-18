"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""
from pathlib import Path

from datasets.cqt_midi_salience_dataset import CqtMidiSalienceDataset
from datasets.create_cqt_store import create_cqt_store


def convert_maestro(
        output_dir: str,
        n_samples: int = 10,
        duration: float = 20,
        label_type: str = "power",
        samples_per_shard: int = 1024,
        num_workers: int = 1,
) -> None:
    dataset = CqtMidiSalienceDataset(
        wav_midi_path="../../resources/maestro-v3.0.0",
        n_samples=n_samples,
        duration=duration,
        label_type=label_type,
        frame_rate=10,
    )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    create_cqt_store(
        dataset,
        output_path,
        samples_per_shard=samples_per_shard,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    convert_maestro(
        output_dir=f"../../resources/maestro_cqts/maestro_activation_5000sam_20sec",
        n_samples=5000,
        samples_per_shard=500,
        duration=20,
        label_type="activation",
        num_workers=8,
    )
