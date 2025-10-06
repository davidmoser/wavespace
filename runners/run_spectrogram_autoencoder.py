
from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.train import single_run

single_run(Configuration(
    spec_file="../resources/logspectrograms.pt",
    epochs=10,
    batch=32,
    base_ch=4,
    version="v5",
    lr=1e-2,
    lr_decay=0.9,
    save_model=False,
))
