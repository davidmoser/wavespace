import subprocess
from typing import Any

import runpod

import pitch_detection_auto.train as train
from pitch_detection_auto.configuration import Configuration


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
        train.single_run(cfg)
        return "Run finished"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
