import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from pitch_detection import train_initial_weights
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


def log_epoch_sample(model, spec_tensor):
    """Log original spec, f0 map and reconstruction as one image."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = spec_tensor.unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,F,T)
        y, f = model(x)

    x_np = x.squeeze().cpu().numpy()
    f_np = f.sum(dim=1).squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()

    fig, ax = plt.subplots(3, 1, figsize=(6, 7), constrained_layout=True)
    for im, title, a in zip((x_np, f_np, y_np), ("original", "f0", "output"), ax):
        a.imshow(im, aspect="auto", origin="lower", cmap="coolwarm")
        a.set_title(title)
        a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)]})
    plt.close(fig)


def train(cfg: Configuration):
    if cfg.train_initial_weights:
        train_initial_weights.train(cfg)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=0)

    vis_spec = data[0].to(dev)

    model = PitchAutoencoder(base_ch=cfg.base_ch, out_ch=cfg.out_ch, kernel_len=cfg.kernel_len).to(dev)
    if cfg.initial_weights_file is not None:
        model.pitch_det_net.load_state_dict(torch.load(cfg.initial_weights_file, map_location=dev))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    if wandb.run:
        log_epoch_sample(model, vis_spec)

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
                wandb.log({"loss": loss.item()}, step=step)

        print("\n")
        print(f"Epoch {epoch:3d}: L={tot / cnt:.4f}")

        if wandb.run:
            log_epoch_sample(model, vis_spec)

        sch.step()

    if cfg.save_model:
        torch.save(model.state_dict(), cfg.save_file)
        print(f"Model saved → {cfg.save_file}")

    wandb.finish()
    print("Done")
