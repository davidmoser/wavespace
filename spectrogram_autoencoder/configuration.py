import json
import sys
from dataclasses import dataclass


@dataclass
class Configuration:
    # model / training
    version: str = "v4"
    epochs: int = 50
    batch: int = 16
    lr: float = 1e-3
    # data
    spec_file: str = "../resources/spectrograms.pt"
    ckpt_dir: str = "../resources/checkpoints"
    # runtime
    save_model: bool = True


def load_config() -> Configuration:
    """Read a JSON object from STDIN and build a WsConfiguration."""
    raw: str = sys.stdin.read()
    if not raw.strip():  # empty ⇒ defaults
        return Configuration()
    cfg = json.loads(raw)
    return Configuration(**cfg)
