import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Configuration:
    # saving
    save: bool = True
    save_file: str = "checkpoints/pitch_head.pt"

    # data and loader
    batch_size: int = 64
    num_workers: int = 4
    sample_duration: float = 1.0
    seq_len: int = 75
    latent_dim: int = 128
    transpose_labels: bool = False

    # optimization
    epochs: Optional[int] = None
    steps: Optional[int] = None
    lr: float = 3e-4
    weight_decay: float = 0.02
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.03

    # device and reproducibility
    device: Optional[str] = None  # None means "cuda if available else cpu"

    # labels / bins
    n_classes: int = 128
    time_frames: int = 75

    # evaluation cadence
    eval_interval: int = 500

    # model
    model_name: str = "DilatedTCN"
    model_config: dict[str, ...] = None

    # dataset
    train_dataset_path: str = None
    val_dataset_path: str | None = None
    split_train_set: Optional[float] = None

