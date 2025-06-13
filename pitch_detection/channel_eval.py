from dataclasses import dataclass
import os
import pathlib
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import wandb

from spectrogram_converter.convert import calculate_log_matrix
from pitch_detection.configuration import Configuration
from pitch_detection.pitch_autoencoder import PitchAutoencoder

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
    ckpt: str
    base_ch: int = 16
    out_ch: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "pitch-channel-eval"


def evaluate_channels(cfg: ChannelEvalConfig) -> Dict[str, List[float]]:
    """Analyse pitch-detector channel activations per instrument."""

    wandb.login(key=os.getenv("WANDB_API_KEY"), anonymous="allow")
    wandb.init(project=cfg.wandb_project, job_type="analysis", config={
        "data_dir": cfg.data_dir,
        "ckpt": cfg.ckpt,
    })

    dev = torch.device(cfg.device)

    df = pd.read_csv(cfg.labels_csv)
    label_for_uuid = dict(zip(df["uuid4"], df["instrument"]))

    data_dir = pathlib.Path(cfg.data_dir)
    wav_files = [p for p in data_dir.glob("*.wav") if not p.name.startswith('.')]
    if not wav_files:
        raise RuntimeError("no wav files found")

    model_cfg = Configuration(spec_file="none", base_ch=cfg.base_ch, out_ch=cfg.out_ch)
    model = PitchAutoencoder(model_cfg).to(dev)
    model.load_state_dict(torch.load(cfg.ckpt, map_location=dev))
    model.eval()

    converter = LogSpectrogram(dev)

    totals_by_instr: Dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(cfg.out_ch, device=dev)
    )
    totals_all = torch.zeros(cfg.out_ch, device=dev)

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
        ax.bar(range(cfg.out_ch), percentages[inst])
        ax.set_title(inst)
        ax.set_ylabel("activation %")
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("channel")
    fig.tight_layout()

    wandb.log({"activations": table, "instrument_plot": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

    return {inst: pct.tolist() for inst, pct in percentages.items()}
