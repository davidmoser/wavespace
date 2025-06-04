import os
import pathlib
from dataclasses import asdict

import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_autoencoder import PitchAutoencoder, entropy_term, laplacian_1d


def sweep_run():
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection")
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection", config=asdict(cfg))
    train(cfg)


def train(cfg: Configuration):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=0)

    model = PitchAutoencoder(cfg.base_ch, 32, cfg.kernel_len).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        cnt = 0
        for spec in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()
            y, f = model(x)  # synth output & f0 activities

            loss = (l1(y, x)
                    + cfg.lambda1 * entropy_term(f)
                    + cfg.lambda2 * f.mean()
                    + cfg.lambda3 * laplacian_1d(f))

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
            cnt += 1
            step += 1
            print(".", end="")
            if wandb.run:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})
        print("\n")
        print(f"Epoch {epoch:3d}: L={tot / cnt:.4f}")
        sch.step()

    if cfg.save_model:
        path = pathlib.Path(cfg.ckpt_dir) / "f0ae.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved → {path}")
