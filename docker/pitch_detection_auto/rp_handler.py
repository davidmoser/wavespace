import subprocess
import sys
from typing import Any

import runpod

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
        run_id = input["run_id"]
        print(f"Single run with id: {run_id}")
        rc = subprocess.run(
            [sys.executable, "/single_run.py", run_id],
            check=False,
        ).returncode
        return f"exit_code {rc}"


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
