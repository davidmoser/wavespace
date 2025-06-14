import os
import pathlib
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.models import get_model


def sweep_run():
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="spectrogram-autoencoder")
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="spectrogram-autoencoder", config=asdict(cfg))
    train(cfg)


def log_epoch_sample(model, spec_tensor, step):
    """Log original spec and reconstruction as one image."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = spec_tensor.unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,F,T)
        y = model(x)

    x_np = x.squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(6, 7), constrained_layout=True)
    for im, title, a in zip((x_np, y_np), ("original", "output"), ax):
        a.imshow(im, aspect="auto", origin="lower", cmap="coolwarm")
        a.set_title(title)
        a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)]}, step=step)
    plt.close(fig)


def train(cfg: Configuration) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch,
                        shuffle=True, pin_memory=False, num_workers=0)

    vis_spec = data[0].to(dev)

    model = get_model(cfg.version, cfg.base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    num_samples = len(loader) * cfg.batch
    print(f"Training {cfg.version}: {cfg.epochs} epochs, {num_samples} samples")

    log_epoch_sample(model, vis_spec, step=0)

    step = 0
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
            step += 1

            wandb.log({"loss": loss.item(), "epoch": epoch}, step=step)
            log_epoch_sample(model, vis_spec, step=step)
            print(".", end="")

        print(f"\nL={total / count:.4f}")
        scheduler.step()

    if cfg.save_model:
        ckpt_dir = pathlib.Path(cfg.ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"spec_auto_{cfg.version}.pt"
        print(f"Saving model to {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()
    print("Done")
