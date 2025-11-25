from dataclasses import dataclass


@dataclass
class Configuration:
    dataset_path: str = None
    num_workers: int = 0
    steps: int = 50
    eval_interval: int = 10
    batch_size: int = 32
    base_ch: int = 16
    out_ch: int = 32
    lr: float = 1e-3
    pitch_det_lr: float | None = None
    kernel_f_len: int = 128
    kernel_t_len: int = 1
    kernel_random: bool = False
    kernel_value: float = 0.1
    force_f0: bool = False
    init_f0: str = "none"  # options: "none", "point", "exponential"
    lambda1: float = 1e-3  # dual variable learning rate
    lambda2: float = 0.6  # \u03b1 in constraint H(q) <= \u03b1 H(p)
    lambda_init: float = 0.0  # initial dual variable value
    train_pitch_det_only: bool = True
    pitch_det_file: str = None
    pitch_autoenc_file: str = None
    save_model: bool = True
    save_file: str = None
    pitch_det_version: str = "v2"
    synth_net_version: str = "v1"
