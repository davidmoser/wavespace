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

    # optimization
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.02
    max_grad_norm: float = 1.0

    # schedule
    warmup_steps: int = 1000
    total_steps_override: Optional[int] = None

    # device and reproducibility
    device: Optional[str] = None  # None means "cuda if available else cpu"

    # labels / bins
    n_classes: int = 128
    fmin_hz: float = 55.0
    fmax_hz: float = 1760.0
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

    def centers_hz(self) -> List[float]:
        log_min = math.log(self.fmin_hz)
        log_max = math.log(self.fmax_hz)
        step = (log_max - log_min) / (self.n_classes - 1)
        return [math.exp(log_min + step * i) for i in range(self.n_classes)]
