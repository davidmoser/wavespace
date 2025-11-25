import matplotlib.pyplot as plt
import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from datasets.tensor_store import TensorStore
from pitch_detection_auto.configuration import Configuration
from pitch_detection_auto.pitch_autoencoder import get_pitch_det_model


def log_sample(model, spec_tensor, current_step):
    """Log original spec, reconstruction as one image."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = spec_tensor.unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,F,T)
        y = torch.sigmoid(model(x))

    x_np = x.squeeze().cpu().numpy()
    y_np = y[:, 0].squeeze().cpu().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(6, 7), constrained_layout=True)
    for im, title, a in zip((x_np, y_np), ("original", "output"), ax):
        a.imshow(im, aspect="auto", origin="lower", cmap="viridis")
        a.set_title(title)
        a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)]}, step=current_step)
    plt.close(fig)


def train(cfg: Configuration) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    dataset = TensorStore(cfg.dataset_path, transpose_samples=True, sample_property="cqts")
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, pin_memory=False, num_workers=0)

    vis_spec = normalize(dataset[0][0].to(dev).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    model = get_pitch_det_model(version=cfg.pitch_det_version, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(opt, gamma=1 - cfg.lr_decay)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(1).to(dev))

    print(f"Training initial weights: {cfg.steps} steps, {cfg.batch_size} batch size")

    current_step = 0
    model.train()
    epoch = 1
    while True:
        print(f"Epoch {epoch}")
        for spec, _ in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()  # (B, C=1, F, T)
            x = normalize(x)
            y = model(x)  # (B, C=32, F, T)
            target = x.repeat(1, cfg.out_ch, 1, 1)  # (B, C=32, F, T)
            loss = criterion(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            current_step += 1
            scheduler.step()

            if current_step >= cfg.steps:
                break

            wandb.log({"loss": loss.item()}, step=current_step)
            if current_step % cfg.eval_interval == 0:
                print(f"Step {current_step}")
                model.eval()
                log_sample(model, vis_spec, current_step)
                model.train()

        model.eval()
        log_sample(model, vis_spec, current_step)
        model.train()
        epoch += 1
        if current_step >= cfg.steps:
            break

    if cfg.save_model:
        print(f"Saving model to {cfg.pitch_det_file}")
        torch.save(model.state_dict(), cfg.pitch_det_file)

    print("Done")


def normalize(samples: torch.Tensor) -> torch.Tensor:
    # samples = torch.log(samples)
    mins = samples.amin(dim=(2,), keepdim=True)
    maxs = samples.amax(dim=(2,), keepdim=True)
    samples = (samples - mins) / (maxs - mins + 1e-4)
    return samples
