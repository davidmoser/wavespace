import subprocess
from typing import Any

import runpod

from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.train_autoencoder import single_run


def handler(event: dict[str, Any]) -> str:
    print(f"Worker Start")
    input = event["input"]
    if "sweep_id" in input:
        sweep_id = input["sweep_id"]
        print(f"Runs with sweep id: {sweep_id}")
        args = ["wandb", "agent", sweep_id]
        if "count" in input:
            count = input["count"]
            args.extend(["--count", count])
        rc = subprocess.run(args).returncode
        return f"exit_code {rc}"
    else:
        cfg = Configuration(**input)
        print(f"Single run with configuration: {cfg}")
        single_run(cfg)
        return "Run finished"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
