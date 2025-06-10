from dataclasses import dataclass


@dataclass
class Configuration:
    # model / training
    batch: int = 16
    sr: int = 22_050  # sample-rate
    dur: float = 4.0  # crop length in seconds
    type: str = "mel"  # "mel" or "log"
    power: float = 1.0
    log_power: bool = False
    # data
    audio_dir: str = "../resources/Medley-solos-DB"
    spec_file: str = "../resources/melspectrograms.pt"
    # runtime
    num_workers: int = 4
