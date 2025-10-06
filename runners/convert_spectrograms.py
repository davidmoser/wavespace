from spectrogram_converter.configuration import Configuration
from spectrogram_converter.convert import convert

convert(Configuration(
    audio_dir="../resources/Medley-solos-DB-sample",
    num_workers=0,
    spec_file="../resources/logspectrograms.pt",
    type="log",
    power=1.0,
    log_power=False,
))
