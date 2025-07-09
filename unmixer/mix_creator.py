from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np

__all__ = [
    "create_room_configs",
    "generate_mixes",
]


def create_room_configs() -> list[dict]:
    return [
        {"name": "small_absorptive", "dimensions": (5.0, 4.0, 3.0), "absorption": 0.6, "max_order": 3},
        {"name": "small_reflective", "dimensions": (5.0, 4.0, 3.0), "absorption": 0.1, "max_order": 3},
        {"name": "large_absorptive", "dimensions": (15.0, 12.0, 8.0), "absorption": 0.4, "max_order": 3},
        {"name": "large_reflective", "dimensions": (15.0, 12.0, 8.0), "absorption": 0.05, "max_order": 3},
        {"name": "medium_balanced", "dimensions": (9.0, 7.0, 4.0), "absorption": 0.3, "max_order": 3},
    ]


def _choose_dataset(file_lists: dict[str, list[Path]]) -> str:
    names = list(file_lists.keys())
    counts = [len(file_lists[n]) for n in names]
    return random.choices(names, weights=counts, k=1)[0]


def generate_mixes(
    dataset_roots: dict[str, Path],
    output_root: Path,
    *,
    num_mixes: int,
    max_events: int,
    mix_length_s: float,
    target_sr: int,
    random_seed: int | None = None,
) -> None:
    import importlib
    try:
        spatialscaper = importlib.import_module("spatialscaper")
    except ModuleNotFoundError:  # fallback to bundled implementation
        from . import room_scaper as spatialscaper

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    rooms = create_room_configs()
    file_lists = {name: sorted(Path(p).rglob("*.wav")) for name, p in dataset_roots.items()}
    output_root.mkdir(parents=True, exist_ok=True)

    for i in range(num_mixes):
        room_cfg = random.choice(rooms)
        k = random.randint(1, max_events)
        scaper = spatialscaper.SpatialScaper(duration=mix_length_s, sr=target_sr, room_cfg=room_cfg)

        events_info = []
        for _ in range(k):
            ds_name = _choose_dataset(file_lists)
            wav_path = random.choice(file_lists[ds_name])
            t0 = random.uniform(0, mix_length_s)
            scaper.add_source(
                filepath=str(wav_path),
                source_time=t0,
                room_position="random",
                allow_repitch=False,
            )
            events_info.append({"dataset": ds_name, "file": str(wav_path), "start_s": float(t0)})

        base = f"{i:05d}_{k}ev_{room_cfg['name']}_{target_sr}Hz_{int(mix_length_s)}s"
        out_audio = output_root / f"{base}.wav"
        out_meta = output_root / f"{base}.json"
        scaper.generate(str(out_audio), str(out_meta), convolve=True)

        metadata = {"num_events": k, "room": room_cfg, "events": events_info}
        with open(out_meta, "w") as f:
            json.dump(metadata, f, indent=2)

