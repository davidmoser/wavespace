import pathlib

import torch
import torchaudio
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from spectrogram_autoencoder.audio_folder import AudioFolder
from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.models import get_model


def train(cfg: Configuration) -> None:
    dev = cfg.resolved_device

    ds = AudioFolder(pathlib.Path(cfg.audio_dir), cfg.sr, cfg.dur)
    loader = DataLoader(ds, batch_size=cfg.batch,
                        shuffle=True, pin_memory=True, drop_last=True)

    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr, n_fft=4096, hop_length=512,
        n_mels=256, power=2.0).to(dev)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(dev)

    model = get_model(cfg.version).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(opt, gamma=1.0)
    l1 = torch.nn.L1Loss()

    num_samples = len(loader) * cfg.batch
    print(f"Training {cfg.version}: {cfg.epochs} epochs, {num_samples} samples")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        model.train()
        total = 0.0
        count = 0
        for wav in loader:  # (B,T)
            wav = wav.to(dev)
            with torch.no_grad():  # no grads for STFT
                x = to_db(spec(wav)).unsqueeze(1)  # (B,1,F,T)
                T = x.shape[-1] - x.shape[-1] % 16
                x = x[:, :, :, :T]
            y = model(x)
            loss = l1(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            count += 1
            print(".", end="")

        print(f"\nL={total / count:.4f}")
        scheduler.step()

    if cfg.save_model:
        ckpt_dir = pathlib.Path(cfg.ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"spec_auto_{cfg.version}.pt"
        print(f"Saving model to {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

    print("Done")
