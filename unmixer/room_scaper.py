from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

import scaper
import pyroomacoustics as pra

__all__ = ["SpatialScaper"]


class SpatialScaper:
    """Minimal SpatialScaper using scaper and pyroomacoustics."""

    def __init__(self, *, duration: float, sr: int, room_cfg: dict) -> None:
        self.duration = duration
        self.sr = sr
        self.room_cfg = room_cfg
        self._sc = scaper.Scaper(duration, fg_path=".", bg_path=".")
        self._sc.sr = sr
        self._sc.ref_db = -50
        self._events: List[dict] = []

    def add_source(
        self,
        *,
        filepath: str,
        source_time: float,
        room_position: str | tuple[float, float, float] = "random",
        allow_repitch: bool = False,
    ) -> None:
        label = Path(filepath).parent.name
        if label not in self._sc.fg_labels:
            self._sc.fg_labels.append(label)
        # determine duration of file
        with sf.SoundFile(filepath) as f:
            file_dur = f.frames / f.samplerate
        dur = min(file_dur, self.duration - source_time)
        self._sc.add_event(
            ("const", label),
            ("const", filepath),
            ("const", 0.0),
            ("const", source_time),
            ("const", dur),
            ("const", 0),
            ("const", 0),
            ("const", 1),
        )
        self._events.append({"filepath": filepath, "room_position": room_position})

    def generate(self, audio_path: str, meta_path: str, convolve: bool = True) -> None:
        y, jams, ann_list, event_audio = self._sc.generate()

        room = pra.ShoeBox(
            self.room_cfg["dimensions"],
            fs=self.sr,
            absorption=self.room_cfg.get("absorption", 0.4),
            max_order=self.room_cfg.get("max_order", 3),
        )
        mic_loc = np.array([[d / 2 for d in self.room_cfg["dimensions"]]]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_loc, self.sr))

        for (ann, audio, info) in zip(ann_list, event_audio, self._events):
            start = float(ann[0])
            position = info["room_position"]
            if position == "random":
                dims = self.room_cfg["dimensions"]
                position = [
                    random.uniform(0.5, dims[0] - 0.5),
                    random.uniform(0.5, dims[1] - 0.5),
                    random.uniform(0.5, dims[2] - 0.5),
                ]
            room.add_source(position, signal=audio, delay=start)

        room.compute_rir()
        room.simulate()
        mix = room.mic_array.signals[0]
        sf.write(audio_path, mix, self.sr)

        events_meta = []
        for ann, info in zip(ann_list, self._events):
            events_meta.append(
                {
                    "file": info["filepath"],
                    "start": float(ann[0]),
                    "duration": float(ann[1]),
                    "position": info["room_position"],
                }
            )
        with open(meta_path, "w") as f:
            json.dump({"room": self.room_cfg, "events": events_meta}, f, indent=2)
