import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import wandb

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_autoencoder import PitchAutoencoder
from spectrogram_converter.convert import calculate_log_matrix

SR = 22_050
N_FFT = 4096
HOP = 512
N_BINS = 256


class LogSpectrogram:
    """Convert waveform tensors to log-frequency magnitude spectrograms."""

    def __init__(self, device: torch.device):
        self.device = device
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT, hop_length=HOP, power=1.0
        ).to(device)
        self.W = calculate_log_matrix(N_FFT, SR, N_BINS).to(device)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device)
        mag = self.spec(wav)
        return torch.matmul(self.W, mag[0]).unsqueeze(0)


def wav_to_spec(path: pathlib.Path, converter: LogSpectrogram) -> torch.Tensor:
    wav, sr0 = torchaudio.load(path)
    wav = wav.mean(0, keepdim=True)
    if sr0 != SR:
        wav = torchaudio.functional.resample(wav, sr0, SR)
    with torch.no_grad():
        return converter(wav)


@dataclass
class ChannelEvalConfig:
    data_dir: str
    labels_csv: str
    model_cfg: Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "pitch-channel-eval"

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelEvalConfig":
        if "model_cfg" in data:
            data["model_cfg"] = Configuration(**data["model_cfg"])
        return cls(**data)


def evaluate_channels(cfg: ChannelEvalConfig) -> dict[str, list[float]]:
    """Analyse pitch-detector channel activations per instrument."""

    wandb.login(key=os.getenv("WANDB_API_KEY"), anonymous="allow")
    wandb.init(project=cfg.wandb_project, job_type="analysis", config=asdict(cfg))

    dev = torch.device(cfg.device)

    df = pd.read_csv(cfg.labels_csv)
    label_for_uuid = dict(zip(df["uuid4"], df["instrument"]))

    data_dir = pathlib.Path(cfg.data_dir)
    wav_files = [p for p in data_dir.glob("*.wav") if not p.name.startswith('.')]
    if not wav_files:
        raise RuntimeError("no wav files found")

    model = PitchAutoencoder(cfg.model_cfg).to(dev)
    model.load_state_dict(torch.load(cfg.model_cfg.pitch_autoenc_file, map_location=dev))
    model.eval()

    converter = LogSpectrogram(dev)

    totals_by_instr: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(cfg.model_cfg.out_ch, device=dev)
    )
    totals_all = torch.zeros(cfg.model_cfg.out_ch, device=dev)

    for wav_path in wav_files:
        uuid = wav_path.stem.split("_")[-1]
        instr = label_for_uuid.get(uuid)
        if instr is None:
            print(f"[warn] uuid {uuid} missing in CSV – skipped")
            continue

        spec = wav_to_spec(wav_path, converter).to(dev)
        with torch.no_grad():
            act = model.pitch_det_net(spec.unsqueeze(0))
        act_sum = act.sum(dim=(2, 3)).squeeze(0)

        totals_by_instr[instr] += act_sum
        totals_all += act_sum

    instruments = sorted(totals_by_instr.keys())
    percentages = {
        inst: (totals_by_instr[inst] / totals_all).cpu()
        for inst in instruments
    }

    rows = []
    for inst in instruments:
        for ch, (act, pct) in enumerate(zip(totals_by_instr[inst].cpu(), percentages[inst])):
            rows.append([inst, ch, act.item(), pct.item()])

    table = wandb.Table(
        columns=["instrument", "channel", "activation", "percentage"], data=rows
    )

    fig, axes = plt.subplots(len(instruments), 1, figsize=(8, 3 * len(instruments)), sharex=True)
    if len(instruments) == 1:
        axes = [axes]
    for ax, inst in zip(axes, instruments):
        ax.bar(range(cfg.model_cfg.out_ch), percentages[inst])
        ax.set_title(inst)
        ax.set_ylabel("activation %")
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("channel")
    fig.tight_layout()

    wandb.log({"activations": table, "instrument_plot": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

    return {inst: pct.tolist() for inst, pct in percentages.items()}
