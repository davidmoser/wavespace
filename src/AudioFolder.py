import pathlib
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioFolder(Dataset):
    """Return mono, resampled waveforms"""

    def __init__(self, root: pathlib.Path, sample_rate: int, duration: float,
                 exts=(".wav", ".flac", ".mp3", ".ogg")):
        self.files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
        self.sr = sample_rate
        self.win = int(duration * sample_rate) if duration else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        wav, sr = torchaudio.load(self.files[idx])  # (C,T)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
        if self.win:  # fixed-length crop / pad
            if wav.shape[1] >= self.win:
                start = random.randint(0, wav.shape[1] - self.win)
                wav = wav[:, start:start + self.win]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.win - wav.shape[1]))
        return wav.squeeze(0)
