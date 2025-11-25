import matplotlib.pyplot as plt
import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets.tensor_store import TensorStore
from pitch_detection_auto.configuration import Configuration
from pitch_detection_auto.evaluate import log_pitch_det_sample
from pitch_detection_auto.pitch_autoencoder import get_pitch_det_model
from pitch_detection_auto.utils import normalize_samples




def train(cfg: Configuration) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    dataset = TensorStore(cfg.dataset_path, transpose_samples=True, sample_property="cqts")
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, pin_memory=False, num_workers=0)

    vis_spec = normalize_samples(dataset[0][0].to(dev).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    model = get_pitch_det_model(version=cfg.pitch_det_version, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(opt, cfg.steps)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(1).to(dev))

    print(f"Training initial weights: {cfg.steps} steps, {cfg.batch_size} batch size")

    current_step = 0
    model.train()
    epoch = 1
    while True:
        print(f"Epoch {epoch}")
        for spec, _ in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()  # (B, C=1, F, T)
            x = normalize_samples(x)
            y = model(x)  # (B, C=32, F, T)
            target = x.repeat(1, cfg.out_ch, 1, 1)  # (B, C=32, F, T)
            loss = criterion(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            current_step += 1
            scheduler.step()

            wandb.log({"loss": loss.item()}, step=current_step)
            if current_step % cfg.eval_interval == 0:
                print(f"Step {current_step}")
                model.eval()
                log_pitch_det_sample(model, vis_spec, current_step)
                model.train()

            if current_step >= cfg.steps:
                break

        epoch += 1
        if current_step >= cfg.steps:
            break

    if cfg.save_model:
        print(f"Saving model to {cfg.pitch_det_file}")
        torch.save(model.state_dict(), cfg.pitch_det_file)

    print("Done")
