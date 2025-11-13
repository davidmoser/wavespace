from utils import run_wandb_run


run_wandb_run(
    config_path="../run_configs/pitch_detection_supervised_power/run_8.yaml",
    project="pitch-detection-supervised",
    run_namespace="david-moser-ggg/pitch-detection-supervised",
    endpoint="9hr07oet4wfndt",
    is_runpod=False,
)
