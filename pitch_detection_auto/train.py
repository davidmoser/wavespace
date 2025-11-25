from dataclasses import asdict

import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets.tensor_store import TensorStore
from pitch_detection_auto import train_pitch_net
from pitch_detection_auto.configuration import Configuration
from pitch_detection_auto.evaluate import log_autoencoder_sample, log_synth_kernels
from pitch_detection_auto.pitch_autoencoder import PitchAutoencoder, entropy_term
from pitch_detection_auto.utils import normalize_samples
from utils.wandb_basic import login_to_wandb

PROJECT_NAME = "pitch-detection-auto"


def sweep_run():
    login_to_wandb()
    wandb.init(project=PROJECT_NAME)
    cfg = wandb.config
    train(Configuration(**cfg.as_dict()))


def single_run(cfg: Configuration):
    login_to_wandb()
    wandb.init(project=PROJECT_NAME, config=asdict(cfg))
    train(cfg)


def single_run_resume(run_id: str) -> None:
    login_to_wandb()
    wandb.init(project=PROJECT_NAME, id=run_id, resume="must")
    cfg = Configuration(**wandb.config.as_dict())
    train(cfg)


def train(cfg: Configuration):
    if cfg.train_pitch_det_only:
        train_pitch_net.train(cfg)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    dataset = TensorStore(cfg.dataset_path, transpose_samples=True, sample_property="cqts")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    sample_specs, _ = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    normalize_samples(sample_specs)

    model = PitchAutoencoder(cfg=cfg).to(dev)
    # load either just the pitch_det_net or the whole pitch_autoencoder (pitch_det + synth_net)
    if cfg.pitch_det_file is not None:
        model.pitch_det_net.load_state_dict(torch.load(cfg.pitch_det_file, map_location=dev))
    if cfg.pitch_autoenc_file is not None:
        model.load_state_dict(torch.load(cfg.pitch_autoenc_file, map_location=dev))

    if cfg.pitch_det_lr is not None:
        opt = torch.optim.AdamW(
            [
                {"params": model.synth.parameters(), "lr": cfg.lr},
                {"params": model.pitch_det_net.parameters(), "lr": cfg.pitch_det_lr},
            ]
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    scheduler = CosineAnnealingLR(opt, cfg.steps)
    l0 = BCEWithLogitsLoss()

    if wandb.run:
        log_autoencoder_sample(model, sample_specs, step=0)
        log_synth_kernels(model, step=0, hide_f0=cfg.init_f0)

    lam = cfg.lambda_init  # dual variable for entropy constraint
    current_step = 0
    model.train()
    epoch = 1
    while True:
        print(f"Epoch {epoch}")
        for spec, _ in loader:  # (B,F,T)
            x = spec.to(dev).unsqueeze(1).float()  # (B,1,F,T)
            x = normalize_samples(x)
            y, f = model(x)  # synth output & f0 activities, (B,1,F,T), (B,C,F,T)

            loss0 = l0(y, x)
            hx = entropy_term(x).mean()
            hf = entropy_term(f).mean()
            loss1 = hf - hx + cfg.lambda2
            loss = loss0 + lam * loss1
            opt.zero_grad()
            loss.backward()
            opt.step()
            current_step += 1
            scheduler.step()

            if current_step >= cfg.steps:
                break

            # dual ascent step
            lam = max(0.0, lam + cfg.lambda1 * loss1.item())

            wandb.log({"loss": loss.item(), "loss0": loss0, "loss1": loss1, "lam": lam}, step=current_step)

            if current_step % cfg.eval_interval == 0:
                print(f"Step {current_step}")
                model.eval()
                log_autoencoder_sample(model, sample_specs, current_step)
                log_synth_kernels(model, step=current_step, hide_f0=cfg.init_f0)
                model.train()

        epoch += 1
        if current_step >= cfg.steps:
            break

    if cfg.save_model:
        torch.save(model.state_dict(), cfg.save_file)
        print(f"Model saved → {cfg.save_file}")

    print("Done")
