from utils import run_wandb_run


run_wandb_run(
    config_path="../../run_configs/pitch_detection_auto/run_train_local.yaml",
    project="pitch-detection-auto",
    endpoint="1a86ns2fgeghvt",
    run_mode="local",
)


# run_wandb_run(
#     config_path="../../run_configs/pitch_detection_supervised_maestro/run_cqt_power_1.yaml",
#     project="pitch-detection-supervised",
#     endpoint="9hr07oet4wfndt",
#     run_mode="runpod",
# )
