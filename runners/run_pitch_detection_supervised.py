from pitch_detection_supervised.configuration import Configuration
from pitch_detection_supervised.train import single_run

single_run(Configuration(
    # save
    save=True,
    save_file="../resources/checkpoints/pitch_detection_supervised/token_transformer_activation_3000.pt",
    # data and loader
    batch_size=16,
    num_workers=1,
    seq_len=150,
    sample_duration=2.0,

    # optimization
    epochs=None,
    steps=3000,
    lr=1e-3,
    weight_decay=0.02,
    max_grad_norm=1.0,
    warmup_fraction=0.03,

    # device and reproducibility
    device="cpu",  # None means "cuda if available else cpu"

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
    train_dataset_path="../resources/encodec_latents/poly_async_activation",
    val_dataset_path=None,
    split_train_set=0.1
))
