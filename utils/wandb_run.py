import csv
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping, Sequence

import requests
import wandb
import yaml


def _format_values(value: Any, format_kwargs: dict[str, Any]) -> Any:
    if isinstance(value, MutableMapping):
        return {k: _format_values(v, format_kwargs) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_format_values(item, format_kwargs) for item in value]
    if isinstance(value, str):
        try:
            return value.format(**format_kwargs)
        except KeyError:
            return value
    return value


VOLUMES = {
    "local": "../../resources",
    "docker": "../resources",
    "runpod": "/runpod-volume",
}


def run_wandb_run(
        config_path: str,
        *,
        project: str,
        run_namespace: str,
        endpoint: str,
        run_mode: str = "runpod",
):
    """Launch a single W&B run and trigger the corresponding RunPod job."""

    if not run_mode in VOLUMES:
        raise ValueError(f"Invalid run mode: {run_mode}")

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config_data: dict[str, Any] = yaml.safe_load(handle)

    format_kwargs = {"volume": VOLUMES[run_mode]}
    config_data = _format_values(config_data, format_kwargs)

    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=wandb_api_key)

    run = wandb.init(project=project, config=config_data)

    run_id = run.id
    run_url = run.url or f"https://wandb.ai/{run_namespace}/runs/{run_id}"

    _record_run_details(
        run_id=run_id,
        config_directory=config_file.parent,
        config_filename=config_file.name,
    )

    if run_mode == "runpod":
        runpod_api_key = os.environ["RUNPOD_API_KEY"]
        headers = {"Authorization": f"Bearer {runpod_api_key}"}
        payload = {"input": {"run_id": run_id}}

        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint}/run",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Job ID: {job_id}")
    elif run_mode == "local":
        # Dynamically import the single_run_resume method, a bit fragile, but better than circular package imports
        module_path = project.replace("-", "_") + ".train"
        module = importlib.import_module(module_path)
        single_run_resume = getattr(module, "single_run_resume")
        single_run_resume(run_id)
    elif run_mode == "docker":
        path = "../../docker/pitch_detection_supervised/test_input.json"
        with open(path) as f:
            data = json.load(f)
        data["input"]["run_id"] = run_id
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _record_run_details(*, run_id: str, config_directory: Path, config_filename: str) -> None:
    runs_file = config_directory / "runs.csv"
    write_header = not runs_file.exists()

    runs_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    with runs_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["file", "id", "timestamp"])
        writer.writerow([config_filename, run_id, timestamp])
