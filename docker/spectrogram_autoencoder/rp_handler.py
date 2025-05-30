import os

import runpod
import wandb

from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.train import sweep_run, single_run


def handler(event):
    print(f"Worker Start")
    input = event['input']
    wandb.login(key=os.environ['WANDB_API_KEY'], verify=True)
    if "sweep_id" in input:
        sweep_id = input["sweep_id"]
        print(f"Runs with sweep id: {sweep_id}")
        sweep_run(sweep_id)
        return "Runs completed"
    else:
        cfg = Configuration(**input)
        print(f"Single run with configuration: {cfg}")
        single_run(cfg)
        return "Run finished"


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
