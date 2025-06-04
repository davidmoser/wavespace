import json
import sys
from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    # model / training
    batch: int = 16
    sr: int = 22_050  # sample-rate
    dur: float = 4.0  # crop length in seconds
    type: str = "mel" # "mel" or "log"
    # data
    audio_dir: str = "../resources/Medley-solos-DB"
    spec_file: str = "../resources/melspectrograms.pt"
    # runtime
    device: str = "auto"  # "auto" → choose cuda if available, else cpu
    num_workers: int = 4

    # helper so we don’t repeat the decision everywhere
    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def load_config() -> Configuration:
    """Read a JSON object from STDIN and build a WsConfiguration."""
    raw: str = sys.stdin.read()
    if not raw.strip():  # empty ⇒ defaults
        return Configuration()
    cfg = json.loads(raw)
    return Configuration(**cfg)
