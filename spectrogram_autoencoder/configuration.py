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
    lr_decay: float = 0.6
    base_ch: int = 4
    # data
    spec_file: str = "../resources/melspectrograms.pt"
    ckpt_dir: str = "../resources/checkpoints"
    # runtime
    save_model: bool = True