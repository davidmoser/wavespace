import os
from typing import Optional

import requests
import wandb

ENDPOINT = "1a86ns2fgeghvt"  # RunPod endpoint ID
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]  # auth for REST calls

is_runpod = True
volume = "/runpod-volume" if is_runpod else "../resources"

sweep_cfg = {
    "program": "sweep_run.py",
    "method": "random",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "run_cap": 1,
    "parameters": {
        "save": {"value": True},
        "save_file": {"value": f"{volume}/checkpoints/pitch_detection_supervised.pt"},
        "batch_size": {"value": 64},
        "num_workers": {"value": 4},
        "sample_duration": {"value": 1.0},
        "seq_len": {"value": 75},
        "latent_dim": {"value": 128},
        "epochs": {"value": 50},
        "lr": {"value": 3e-4},
        "weight_decay": {"value": 0.02},
        "max_grad_norm": {"value": 1.0},
        "warmup_steps": {"value": 1000},
        "device": {"value": "cuda"},
        "n_classes": {"value": 128},
        "fmin_hz": {"value": 55.0},
        "fmax_hz": {"value": 1760.0},
        "time_frames": {"value": 75},
        "log_interval": {"value": 50},
        "eval_interval": {"value": 500},
        "model_name": {"value": "DilatedTCN"},
        "model_config": {"value": {"seq_len": 75, "latent_dim": 128, "n_classes": 128}},
        "train_dataset_path": {"value": f"{volume}/encodec_latents/poly_async_1"},
        "val_dataset_path": {"value": None},
    },
}


def launch_sweep() -> str:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="pitch-detection-supervised")
    return sweep_id


def trigger_runpod_job(sweep_id: str) -> Optional[str]:
    if not is_runpod:
        return None
    payload = {"input": {"sweep_id": f"david-moser-ggg/pitch-detection-supervised/{sweep_id}"}}
    headers = {"Authorization": f"Bearer {RUNPOD_KEY}"}
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT}/run",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    job_id = response.json()["id"]
    return job_id


def main() -> None:
    sweep_id = launch_sweep()
    job_id = trigger_runpod_job(sweep_id)
    if job_id is not None:
        print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
