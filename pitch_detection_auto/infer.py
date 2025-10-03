import pathlib

import matplotlib.pyplot as plt
import torch

from pitch_detection_auto.configuration import Configuration as PDConfig
from pitch_detection_auto.pitch_autoencoder import PitchAutoencoder
from spectrogram_converter.configuration import Configuration as ConvertConfig
from spectrogram_converter.convert import convert


def infer(
        convert_cfg: ConvertConfig,
        pitch_net_cfg: PDConfig,
        pitch_autoencoder_file: str,
) -> None:
    specs = convert(convert_cfg)  # (N,F,T)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PitchAutoencoder(pitch_net_cfg).to(device)
    model.load_state_dict(torch.load(pitch_autoencoder_file, map_location=device))
    model.eval()

    for idx, spec in enumerate(specs):
        with torch.no_grad():
            x = spec.unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,F,T)
            y, f = model(x)

        x_np = x.squeeze().cpu().numpy()
        f_np = f.sum(dim=1).squeeze().cpu().numpy()  # (1,F,T)
        # censor the cheat
        f_np[-3:, -3:] = torch.zeros(3, 3)
        y_np = y.squeeze().cpu().numpy()

        fig, ax = plt.subplots(3, 1, figsize=(6, 7), constrained_layout=True)
        for data, title, axis in zip((x_np, f_np, y_np), ("original", "f0", "output"), ax):
            axis.imshow(data, aspect="auto", origin="lower", cmap="coolwarm")
            axis.set_title(title)
            axis.axis("off")

        out_img = pathlib.Path(convert_cfg.audio_dir) / f"autoencoded_{idx}.png"
        plt.savefig(out_img)
        plt.close(fig)
