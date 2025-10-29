import os
from typing import Optional

import requests
import wandb

ENDPOINT = "9hr07oet4wfndt"  # RunPod endpoint ID
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]  # auth for REST calls

is_runpod = True
volume = "/runpod-volume" if is_runpod else "../resources"
num_workers = 5

# {"min": 0, "max": 0.1, "distribution": "uniform"}
# {"min": 1e-4, "max": 1e-3, "distribution": "log_uniform_values"}

sweep_cfg = {
    "program": "sweep_run.py",
    "method": "random",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "run_cap": 1,
    "parameters": {
        "save": {"value": False},
        "save_file": {"value": f"{volume}/checkpoints/pitch_detection_supervised.pt"},
        "batch_size": {"value": 128},
        "num_workers": {"value": 8},
        "seq_len": {"value": 150},
        "sample_duration": {"value": 2.0},
        "steps": {"value": 1000},
        "lr": {"value": 0.004},
        "weight_decay": {"value": 0.1},
        "warmup_fraction": {"value": 0.03},
        "device": {"value": "cuda"},
        "eval_interval": {"value": 30},
        "model_name": {"value": "TokenTransformer"},
        "model_config": {
            "parameters": {
                "seq_len": {"value": 150},
                "dropout": {"value": 0.2}
            }
        },
        "train_dataset_path": {"value": f"{volume}/encodec_latents/poly_async_activation"},
        "split_train_set": {"value": 0.1},
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
    for _ in range(min(num_workers, sweep_cfg.get("run_cap"))):
        job_id = trigger_runpod_job(sweep_id)
        if job_id is not None:
            print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
