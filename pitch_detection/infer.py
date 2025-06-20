import pathlib
import torch
import matplotlib.pyplot as plt

from spectrogram_converter.configuration import Configuration as ConvertConfig
from spectrogram_converter.convert import convert
from pitch_detection.configuration import Configuration as PDConfig
from pitch_detection.pitch_autoencoder import PitchAutoencoder


def infer(audio_folder: str,
          configuration: ConvertConfig,
          pitch_autoencoder_file: str,
          pitch_det_version: str,
          synth_net_version: str) -> None:
    """Convert audio files and run the pitch autoencoder on them.

    Parameters
    ----------
    audio_folder: str
        Directory containing audio files to analyse.
    configuration: ConvertConfig
        Conversion settings for spectrogram extraction.
    pitch_autoencoder_file: str
        Path to the weight file of the pitch autoencoder.
    pitch_det_version: str
        Version string for the pitch detector network.
    synth_net_version: str
        Version string for the synthesizer network.
    """
    configuration.audio_dir = audio_folder
    configuration.spec_file = None  # no file output

    specs = convert(configuration)  # (N,F,T)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = PDConfig(
        spec_file="none",
        pitch_det_version=pitch_det_version,
        synth_net_version=synth_net_version,
    )
    model = PitchAutoencoder(model_cfg).to(device)
    model.load_state_dict(torch.load(pitch_autoencoder_file, map_location=device))
    model.eval()

    for idx, spec in enumerate(specs):
        with torch.no_grad():
            x = spec.unsqueeze(0).unsqueeze(0).float().to(device)
            y, f = model(x)

        x_np = x.squeeze().cpu().numpy()
        f_np = f.sum(dim=1).squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy()

        fig, ax = plt.subplots(3, 1, figsize=(6, 7), constrained_layout=True)
        for data, title, axis in zip((x_np, f_np, y_np), ("original", "f0", "output"), ax):
            axis.imshow(data, aspect="auto", origin="lower", cmap="coolwarm")
            axis.set_title(title)
            axis.axis("off")

        out_img = pathlib.Path(audio_folder) / f"autoencoded_{idx}.png"
        plt.savefig(out_img)
        plt.close(fig)
