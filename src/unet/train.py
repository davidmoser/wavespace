import argparse
import pathlib

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.utils import preprocess
from unet_arch import AudioUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hf-instruments", help="HF dataset id")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda e: e["instrument_id"] < 4)  # keep 4 instruments
    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["mag"])
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, pin_memory=True)

    model = AudioUNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()  # perceptually smoother than MSE

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            x = batch["mag"].to(args.device)  # (B, 1, F, T)
            y = model(x)
            loss = loss_fn(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"epoch {epoch:03d}  L1={total / len(ds):.4f}")

    pathlib.Path("../../resources/checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "../../resources/checkpoints/audio_unet.pt")


if __name__ == "__main__":
    main()
