from pitch_detection_supervised.configuration import Configuration
from pitch_detection_supervised.train import single_run

single_run(Configuration(
    # save
    save=False,
    save_file="checkpoints/pitch_detection_supervised/test.pt",
    # data and loader
    batch_size=16,
    num_workers=1,
    seq_len=150,
    sample_duration=2.0,

    # optimization
    epochs=None,
    steps=1000,
    lr=1e-3,
    weight_decay=0.02,
    max_grad_norm=1.0,
    warmup_steps=100,

    # device and reproducibility
    device="cpu",  # None means "cuda if available else cpu"

    # labels / bins
    n_classes=128,
    fmin_hz=100.0,
    fmax_hz=10000.0,
    time_frames=150,  # matches number of tokens

    # evaluation / logging cadence
    eval_interval=10,

    # model
    # model_name="DilatedTCN",
    # model_config={"seq_len": 150},

    # model_name="LocalContextMLP",
    # model_config={"seq_len": 150},

    model_name="TokenTransformer",
    model_config={"seq_len": 150},

    # dataset
    train_dataset_path="../resources/encodec_latents/poly_async_2",
    val_dataset_path=None,
    split_train_set=0.1
))
