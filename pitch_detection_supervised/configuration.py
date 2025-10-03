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
    use_amp: bool = True

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

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.total_steps_override is not None and self.total_steps_override <= 0:
            raise ValueError("total_steps_override must be positive when provided")
        if self.device is not None and not isinstance(self.device, str):
            raise TypeError("device must be a string or None")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if self.fmin_hz <= 0:
            raise ValueError("fmin_hz must be positive")
        if self.fmax_hz <= self.fmin_hz:
            raise ValueError("fmax_hz must be greater than fmin_hz")
        if self.sigma_bins <= 0:
            raise ValueError("sigma_bins must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.within_bins <= 0:
            raise ValueError("within_bins must be positive")
        if self.centers_explicit is not None:
            if not isinstance(self.centers_explicit, (list, tuple)):
                raise TypeError("centers_explicit must be a sequence of floats")
            centers = list(self.centers_explicit)
            if len(centers) != self.n_classes:
                raise ValueError("centers_explicit must match n_classes in length")
            if any(c <= 0 for c in centers):
                raise ValueError("centers_explicit values must be positive")
            if any(b <= a for a, b in zip(centers, centers[1:])):
                raise ValueError("centers_explicit must be strictly increasing")
            self.centers_explicit = centers

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

    @staticmethod
    def from_dict(d: dict) -> "Config":
        if not isinstance(d, dict):
            raise TypeError("from_dict expects a dictionary")
        valid_keys = {f.name for f in fields(Config)}
        kwargs = {key: value for key, value in d.items() if key in valid_keys}
        config = Config(**kwargs)
        config.validate()
        return config
