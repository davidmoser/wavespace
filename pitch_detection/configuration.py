from dataclasses import dataclass


@dataclass
class Configuration:
    spec_file: str
    epochs: int = 50
    batch: int = 32
    base_ch: int = 16
    lr: float = 1e-3
    lr_decay: float = 0.98
    kernel_len: int = 129
    lambda1: float = 1.0  # entropy
    lambda2: float = 1e-3  # L1 activity
    lambda3: float = 1e-4  # Laplacian
    save_model: bool = True
    ckpt_dir: str = "./checkpoints"
