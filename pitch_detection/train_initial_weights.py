import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_det_net import PitchDetNet


def train(cfg: Configuration) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    data = torch.load(cfg.spec_file, mmap=True, map_location=dev)
    loader = DataLoader(data, batch_size=cfg.batch,
                        shuffle=True, pin_memory=False, num_workers=0)

    model = PitchDetNet(base_ch=cfg.base_ch, out_ch=cfg.out_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(opt, gamma=cfg.lr_decay)
    l1 = torch.nn.L1Loss()

    num_samples = len(loader) * cfg.batch
    print(f"Training initial weights: {cfg.epochs} epochs, {num_samples} samples")

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        model.train()
        total = 0.0
        count = 0
        for spec in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()
            y = model(x)
            target = x.repeat(1, cfg.out_ch, 1, 1)
            loss = l1(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            count += 1
            step += 1

            wandb.log({"loss": loss.item(), "epoch": epoch}, step=step)
            print(".", end="")

        print(f"\nL={total / count:.4f}")
        scheduler.step()

    if cfg.save_model:
        print(f"Saving model to {cfg.save_file}")
        torch.save(model.state_dict(), cfg.save_file)

    print("Done")
