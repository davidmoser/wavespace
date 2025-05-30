import os

import requests
import wandb

from spectrogram_autoencoder.train import sweep_run

ENDPOINT = "idluq2u2vgme12"  # RunPod endpoint ID
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]  # auth for REST calls

sweep_cfg = {
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "run_cap": 3,
    "parameters": {
        "lr": {"min": 1e-4, "max": 1e-2, "distribution": "log_uniform_values"},
        "batch": {"values": [512]},
        "version": {"value": "v4"},
        "epochs": {"value": 1},
        "spec_file": {"value": "/runpod-volume/spectrograms.pt"},
        "save_model": {"value": False},
    }
}

run_local = False

if __name__ == "__main__":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="spectrogram-autoencoder")
    if run_local:
        sweep_run(sweep_id)
    else:
        RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
        payload = {"input": {"sweep_id": sweep_id}}
        headers = {"Authorization": f"Bearer {RUNPOD_KEY}"}
        r = requests.post(f"https://api.runpod.ai/v2/{ENDPOINT}/run",
                          json=payload, headers=headers)
        job_id = r.json()["id"]
        print(f"Job ID: {job_id}")

