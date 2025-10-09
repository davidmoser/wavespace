from dataclasses import dataclass, field, fields
from typing import List, Optional
import math


@dataclass
class Config:
    # saving
    save: bool = True
    save_file: str = "checkpoints/pitch_head.pt"

    # data and loader
    batch_size: int = 64
    num_workers: int = 4
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
    seed: int = 1337

    # labels / bins
    n_classes: int = 128
    fmin_hz: float = 55.0
    fmax_hz: float = 1760.0
    log_bins: bool = True
    centers_explicit: Optional[List[float]] = field(default=None)
    sigma_bins: float = 0.7

    # evaluation / logging cadence
    log_interval: int = 50
    eval_interval: int = 500
    within_bins: int = 1

    def centers_hz(self) -> List[float]:
        if self.centers_explicit is not None:
            if len(self.centers_explicit) != self.n_classes:
                raise ValueError("centers_explicit must match n_classes in length")
            return list(self.centers_explicit)

        if self.n_classes == 1:
            if self.log_bins:
                return [math.sqrt(self.fmin_hz * self.fmax_hz)]
            return [(self.fmin_hz + self.fmax_hz) / 2.0]

        if self.log_bins:
            log_min = math.log(self.fmin_hz)
            log_max = math.log(self.fmax_hz)
            step = (log_max - log_min) / (self.n_classes - 1)
            return [math.exp(log_min + step * i) for i in range(self.n_classes)]

        step = (self.fmax_hz - self.fmin_hz) / (self.n_classes - 1)
        return [self.fmin_hz + step * i for i in range(self.n_classes)]
