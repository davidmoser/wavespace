import matplotlib.pyplot as plt
import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_autoencoder import get_pitch_det_model


def log_epoch_sample(model, spec_tensor):
    """Log original spec, reconstruction as one image."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = spec_tensor.unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,F,T)
        y = model(x)

    x_np = x.squeeze().cpu().numpy()
    y_np = y[:, 0].squeeze().cpu().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(6, 7), constrained_layout=True)
    for im, title, a in zip((x_np, y_np), ("original", "output"), ax):
        a.imshow(im, aspect="auto", origin="lower", cmap="coolwarm")
        a.set_title(title)
        a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)]})
    plt.close(fig)


def train(cfg: Configuration) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch,
                        shuffle=True, pin_memory=False, num_workers=0)

    vis_spec = data[0].to(dev)

    model = get_pitch_det_model(version=cfg.pitch_det_version, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    num_samples = len(loader) * cfg.batch
    print(f"Training initial weights: {cfg.epochs} epochs, {num_samples} samples")

    if wandb.run:
        log_epoch_sample(model, vis_spec)

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        model.train()
        total = 0.0
        count = 0
        for spec in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()  # (B, C=1, F, T)
            y = model(x)  # (B, C=32, F, T)
            target = x.repeat(1, cfg.out_ch, 1, 1)  # (B, C=32, F, T)
            loss = l1(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            count += 1
            step += 1

            wandb.log({"loss": loss.item()}, step=step)
            print(".", end="")

        print(f"\nL={total / count:.4f}")
        if wandb.run:
            log_epoch_sample(model, vis_spec)
        scheduler.step()

    if cfg.save_model:
        print(f"Saving model to {cfg.initial_weights_file}")
        torch.save(model.state_dict(), cfg.initial_weights_file)

    wandb.finish()
    print("Done")
