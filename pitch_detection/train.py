import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from pitch_detection import train_initial_weights
from pitch_detection.configuration import Configuration
from pitch_detection.pitch_autoencoder import PitchAutoencoder, entropy_term


def sweep_run():
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection")
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    wandb.init(project="pitch-detection", config=asdict(cfg))
    train(cfg)


def log_epoch_sample(model, spec_tensor, step):
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

    wandb.log({"epoch_samples": [wandb.Image(fig)], "f0_min": f_np.min(), "f0_max": f_np.max()}, step=step)
    plt.close(fig)


def log_synth_kernels(model, step, hide_f0) -> None:
    """Log SynthNet kernels as a heat-map image.

    Kernels have shape (C,T,F) when T==1 and (C,F,T) when T>1.
    We keep frequency on the y-axis and flatten (channel,time) on x.
    """
    with torch.no_grad():
        ker = F.softplus(model.synth.kernel).cpu().squeeze(1)  # (C,·,·)
        if hide_f0:
            ker = ker[:, 1:]

    if len(ker.shape) == 2:  # (C,F) → 1d case
        img = ker.numpy().T  # (F,C)
        xlabel = "channel"
    elif len(ker.shape) == 3:  # (C,F,T) → 2d case
        C, F_, T = ker.shape
        img = ker.permute(1, 0, 2).reshape(F_, C * T).numpy()  # (F, C·T)
        xlabel = "channel · time"
    else:
        raise ValueError()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title("SynthNet kernels")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    ax.axis("auto")

    wandb.log(
        {"kernels": [wandb.Image(fig)],
         "kernels_min": img.min(),
         "kernels_max": img.max()},
        step=step,
    )
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

    model = PitchAutoencoder(cfg=cfg).to(dev)
    if cfg.initial_weights_file is not None:
        model.pitch_det_net.load_state_dict(torch.load(cfg.initial_weights_file, map_location=dev))

    if cfg.pitch_det_lr is not None:
        opt = torch.optim.AdamW(
            [
                {"params": model.synth.parameters(), "lr": cfg.lr},
                {"params": model.pitch_det_net.parameters(), "lr": cfg.pitch_det_lr},
            ]
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    if wandb.run:
        log_epoch_sample(model, vis_spec, step=0)
        log_synth_kernels(model, step=0, hide_f0=cfg.init_f0)

    step = 0
    lam = cfg.lambda_init  # dual variable for entropy constraint
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        cnt = 0
        for spec in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()  # (B,1,F,T)
            y, f = model(x)  # synth output & f0 activities, (B,1,F,T), (B,C,F,T)

            loss0 = l1(y, x)
            hx = entropy_term(x).mean()
            hf = entropy_term(f).mean()
            loss1 = hf - hx + cfg.lambda2
            loss = loss0 + lam * loss1

            opt.zero_grad()
            loss.backward()
            opt.step()

            # dual ascent step
            lam = max(0.0, lam + cfg.lambda1 * loss1.item())

            tot += loss.item()
            cnt += 1
            step += 1
            print(".", end="")
            if wandb.run:
                wandb.log({"loss": loss.item(), "loss0": loss0, "loss1": loss1, "lam": lam}, step=step)

        print("\n")
        print(f"Epoch {epoch:3d}: L={tot / cnt:.4f}")

        if wandb.run:
            log_epoch_sample(model, vis_spec, step)
            log_synth_kernels(model, step=step, hide_f0=cfg.init_f0)

        sch.step()

    if cfg.save_model:
        torch.save(model.state_dict(), cfg.save_file)
        print(f"Model saved → {cfg.save_file}")

    wandb.finish()
    print("Done")
