
from utils import run_wandb_sweep

# run_wandb_sweep(
#     config_path="../sweep_configs/other/spectrogram_autoencoder.yaml",
#     project="spectrogram-autoencoder",
#     sweep_namespace="david-moser-ggg/spectrogram-autoencoder",
#     endpoint="idluq2u2vgme12",
# )

# run_wandb_sweep(
#     config_path="../sweep_configs/other/pitch_detection_auto.yaml",
#     project="pitch-detection",
#     sweep_namespace="david-moser-ggg/pitch-detection",
#     endpoint="1a86ns2fgeghvt",
# )

run_wandb_sweep(
    config_path="../sweep_configs/pitch_detection_supervised_power/run_7.yaml",
    project="pitch-detection-supervised",
    sweep_namespace="david-moser-ggg/pitch-detection-supervised",
    endpoint="9hr07oet4wfndt",
)
