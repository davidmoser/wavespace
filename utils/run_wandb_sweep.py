from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence

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


def run_wandb_sweep(
    config_path: str,
    *,
    project: str,
    sweep_namespace: str,
    endpoint: str,
    is_runpod: bool = True,
    num_workers: Optional[int] = None,
) -> tuple[str, list[str]]:
    """Launch a W&B sweep and optionally trigger RunPod jobs."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        sweep_cfg: Dict[str, Any] = yaml.safe_load(handle)

    format_kwargs = {"volume": "/runpod-volume" if is_runpod else "../resources"}
    sweep_cfg = _format_values(sweep_cfg, format_kwargs)

    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=wandb_api_key)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project)

    _record_sweep_details(
        sweep_id=sweep_id,
        config_directory=config_file.parent,
        config_filename=config_file.name,
    )

    job_ids: list[str] = []
    if is_runpod:
        runpod_api_key = os.environ["RUNPOD_API_KEY"]
        headers = {"Authorization": f"Bearer {runpod_api_key}"}
        payload = {"input": {"sweep_id": f"{sweep_namespace}/{sweep_id}"}}

        run_cap = sweep_cfg.get("run_cap")
        if num_workers is None:
            if isinstance(run_cap, int) and run_cap > 0:
                max_workers = min(run_cap, 5)
            else:
                max_workers = 1
        else:
            max_workers = max(num_workers, 1)
            if isinstance(run_cap, int) and run_cap > 0:
                max_workers = min(max_workers, run_cap)

        for _ in range(max_workers):
            response = requests.post(
                f"https://api.runpod.ai/v2/{endpoint}/run",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            job_id = response.json()["id"]
            print(f"Job ID: {job_id}")
            job_ids.append(job_id)

    return sweep_id, job_ids


def _record_sweep_details(*, sweep_id: str, config_directory: Path, config_filename: str) -> None:
    sweeps_file = config_directory / "sweeps.csv"
    write_header = not sweeps_file.exists()

    sweeps_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    sweep_hash = sweep_id.split("/")[-1]

    with sweeps_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["id", "file", "timestamp"])
        writer.writerow([sweep_hash, config_filename, timestamp])
