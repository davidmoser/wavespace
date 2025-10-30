from __future__ import annotations

from pathlib import Path

from utils import run_wandb_sweep


if __name__ == "__main__":
    run_wandb_sweep(
        config_path=str(
            Path(__file__).resolve().parent
            / "configs"
            / "spectrogram_autoencoder.yaml"
        ),
        project="spectrogram-autoencoder",
        sweep_namespace="david-moser-ggg/spectrogram-autoencoder",
        endpoint="idluq2u2vgme12",
        is_runpod=True,
    )
