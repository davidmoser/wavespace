import os

import requests
import wandb

ENDPOINT = "idluq2u2vgme12"  # RunPod endpoint ID
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]  # auth for REST calls

is_runpod = True
volume = "/runpod-volume" if is_runpod else "../resources"

sweep_cfg = {
    "program": "sweep_run.py",
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "run_cap": 30,
    "parameters": {
        "lr": {"min": 1e-3, "max": 1e-1, "distribution": "log_uniform_values"},
        "lr_decay": {"values": [0.9]},
        "batch": {"value": 128},
        "version": {"value": "v5"},
        "epochs": {"value": 1},
        "base_ch": {"values": [4, 16]},
        "spec_file": {"value": f"{volume}/spectrograms.pt"},
        "save_model": {"value": False},
    }
}

if __name__ == "__main__":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="spectrogram-autoencoder")

    if is_runpod:
        RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
        payload = {"input": {"sweep_id": f"david-moser-ggg/spectrogram-autoencoder/{sweep_id}"}}
        headers = {"Authorization": f"Bearer {RUNPOD_KEY}"}
        r = requests.post(f"https://api.runpod.ai/v2/{ENDPOINT}/run",
                          json=payload, headers=headers)
        job_id = r.json()["id"]
        print(f"Job ID: {job_id}")
