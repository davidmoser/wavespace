from dataclasses import dataclass


@dataclass
class Configuration:
    spec_file: str
    epochs: int = 50
    batch: int = 32
    base_ch: int = 16
    out_ch: int = 32
    lr: float = 1e-3
    lr_decay: float = 0.98
    kernel_f_len: int = 128
    kernel_t_len: int = 1
    kernel_random: bool = False
    kernel_value: float = 0.1
    force_f0: bool = False
    init_f0: str = "none"  # options: "none", "point", "exponential"
    lambda1: float = 1.0  # entropy
    lambda2: float = 1e-3  # L1 activity
    lambda3: float = 1e-4  # Laplacian
    train_initial_weights: bool = True
    initial_weights_file: str = None
    save_model: bool = True
    save_file: str = None
    pitch_det_version: str = "v2"
    synth_net_version: str = "v1"
