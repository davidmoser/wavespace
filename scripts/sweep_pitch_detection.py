import os

import requests
import wandb

ENDPOINT = "1a86ns2fgeghvt"  # RunPod endpoint ID
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]  # auth for REST calls

is_runpod = True
volume = "/runpod-volume" if is_runpod else "../resources"

sweep_cfg = {
    "program": "sweep_run.py",
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "run_cap": 10,
    "parameters": {
        "lr": {"value": 0.001},
        "lr_decay": {"value": 1.0},
        "pitch_det_lr": {"value": 0.02},
        "lambda_init": {"value": 0.0},
        "lambda1": {"value": 0.002},
        "lambda2": {"min": 1.5, "max": 10, "distribution": "log_uniform_values"},
        "batch": {"value": 128},
        "epochs": {"value": 5},
        "base_ch": {"value": 16},
        "out_ch": {"value": 32},
        "force_f0": {"value": True},
        "init_f0": {"value": "point"},
        "train_pitch_det_only": {"value": False},
        "pitch_autoenc_file": {"value": f"{volume}/checkpoints/pitch_autoencoder_v10_100epochs.pt"},
        "kernel_f_len": {"value": 128},
        "kernel_t_len": {"value": 1},
        "kernel_random": {"value": False},
        "kernel_value": {"value": 0.01},
        "spec_file": {"value": f"{volume}/logspectrograms.pt"},
        "save_model": {"value": True},
        "save_file": {"value": "/runpod-volume/checkpoints/pitch_autoencoder_v10_100epochs_fine.pt"},
        "pitch_det_version": {"value": "v3"},
        "synth_net_version": {"value": "v1"},
    }
}

if __name__ == "__main__":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="pitch-detection")

    if is_runpod:
        RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
        payload = {"input": {"sweep_id": f"david-moser-ggg/pitch-detection/{sweep_id}"}}
        headers = {"Authorization": f"Bearer {RUNPOD_KEY}"}
        r = requests.post(f"https://api.runpod.ai/v2/{ENDPOINT}/run",
                          json=payload, headers=headers)
        job_id = r.json()["id"]
        print(f"Job ID: {job_id}")
