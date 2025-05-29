import pathlib

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.models import get_model


def train(cfg: Configuration) -> None:
    dev = cfg.resolved_device
    print(f"Using device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch,
                        shuffle=True, pin_memory=False, num_workers=0)

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
        for wav in loader:  # (B,F,T)
            x = wav.to(dev).unsqueeze(1).float()
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
