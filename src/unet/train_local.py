import argparse
import pathlib

import torch
import torchaudio
from torch.utils.data import DataLoader

from src.AudioFolder import AudioFolder
from unet_arch import AudioUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", type=pathlib.Path, default="audio",
                   help="folder with raw audio files")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sr", type=int, default=22_050, help="target sample rate")
    p.add_argument("--dur", type=float, default=4.0, help="audio crop length (s)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = AudioFolder(args.audio_dir, args.sr, args.dur)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True)

    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr, n_fft=4096, hop_length=512,
        n_mels=256, power=2.0).to(args.device)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(args.device)

    model = AudioUNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for wav in loader:  # (B,T)
            wav = wav.to(args.device)
            with torch.no_grad():  # no grads for STFT
                x = to_db(spec(wav)).unsqueeze(1)  # (B,1,F,T)
                T = x.shape[-1]
                T = T - T % 8
                x = x[:, :, :, 0:T]
            y = model(x)
            loss = loss_fn(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            print(".", end="")
        print("")
        print(f"epoch {epoch:03d}  L1={total / len(ds):.4f}")

    ckpt_dir = pathlib.Path("../../resources/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "audio_unet.pt")


if __name__ == "__main__":
    main()
