
from pitch_detection_auto.configuration import Configuration
from pitch_detection_auto.train import single_run

single_run(Configuration(
    spec_file="../resources/logspectrograms.pt",
    epochs=5,
    batch=32,
    base_ch=4,
    out_ch=32,
    lr=0.001,
    pitch_det_lr=0.01,
    lr_decay=1,
    kernel_f_len=128,
    kernel_t_len=1,
    kernel_random=False,
    kernel_value=0.005,
    force_f0=False,
    init_f0="exponential",
    lambda1=0.001,  # compression goal learning rate
    lambda2=0.1,  # compression rate
    train_pitch_det_only=False,
    # pitch_autoenc_file="../resources/checkpoints/pitch_det_net_initial_weights_v3.pt",
    save_file="../resources/checkpoints/pitch_autoencoder_v2.pt",
    save_model=False,
    pitch_det_version="v3",
    synth_net_version="v2"
))
