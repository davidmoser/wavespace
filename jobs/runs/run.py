from utils import run_wandb_run


run_wandb_run(
    config_path="../../run_configs/pitch_detection_maestro/run_test.yaml",
    project="pitch-detection-supervised",
    run_namespace="david-moser-ggg/pitch-detection-supervised",
    endpoint="9hr07oet4wfndt",
    run_mode="runpod",
)
