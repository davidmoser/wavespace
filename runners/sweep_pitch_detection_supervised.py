from __future__ import annotations

from pathlib import Path

from utils import run_wandb_sweep


if __name__ == "__main__":
    run_wandb_sweep(
        config_path=str(
            Path(__file__).resolve().parent
            / "configs"
            / "pitch_detection_supervised.yaml"
        ),
        project="pitch-detection-supervised",
        sweep_namespace="david-moser-ggg/pitch-detection-supervised",
        endpoint="9hr07oet4wfndt",
        is_runpod=True,
    )
