import json
import subprocess
import sys
from typing import Any

import runpod

from pitch_detection_supervised.configuration import Configuration

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
        rc = subprocess.run(
            [sys.executable, "/single_run.py", json.dumps(input)],
            check=False,
        ).returncode
        return f"exit_code {rc}"


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
