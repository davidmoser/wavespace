from __future__ import annotations

from pathlib import Path

from utils import run_wandb_sweep


if __name__ == "__main__":
    run_wandb_sweep(
        config_path=str(
            Path(__file__).resolve().parent / "configs" / "pitch_detection_auto.yaml"
        ),
        project="pitch-detection",
        sweep_namespace="david-moser-ggg/pitch-detection",
        endpoint="1a86ns2fgeghvt",
        is_runpod=True,
    )
