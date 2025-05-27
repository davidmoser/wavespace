import argparse
import pathlib
import re

import torch
import torchaudio
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from spec_auto_v1 import SpecAutoNet as NetV1
from spec_auto_v2 import SpecAutoNet as NetV2
from spec_auto_v3 import SpecAutoNet as NetV3
from spec_auto_v4 import SpecAutoNet as NetV4
from src.AudioFolder import AudioFolder

_MODEL_REGISTRY = {
    1: NetV1,
    2: NetV2,
    3: NetV3,
    4: NetV4,
}


def get_model(version: str):
    version = int(re.search(r'v(\d+)_?', version).group(1))
    try:
        return _MODEL_REGISTRY[version]()
    except KeyError:
        raise ValueError(f"Unknown model version {version}")


def main():
    version = "v4"
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", type=pathlib.Path, default="audio",
                   help="folder with raw audio files")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
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

    model = get_model(version).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(opt, gamma=1)

    print(f"Doing {args.epochs} epochs, with {len(loader) * args.batch} samples")
    for epoch in range(1, args.epochs + 1):
        print("\nEpoch {}/{}".format(epoch, args.epochs))
        model.train()
        total = 0.0
        count = 0
        for wav in loader:  # (B,T)
            wav = wav.to(args.device)
            with torch.no_grad():  # no grads for STFT
                x = to_db(spec(wav)).unsqueeze(1)  # (B,1,F,T)
                T = x.shape[-1]
                T = T - T % 16
                x = x[:, :, :, 0:T]
            y = model(x)
            # recon = l1(y, x)
            # gate_l1 = torch.sigmoid(model.alpha1) + torch.sigmoid(model.alpha2) + torch.sigmoid(model.alpha3)
            # loss = recon + lambda_gate * gate_l1
            loss = l1(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            count += 1
            print(".", end="")
            if count % 10 == 0:
                print(f"\nsamples {count * args.batch}  L1={total / 10 / args.batch:.4f}")
                total = 0.0
        scheduler.step()

    ckpt_dir = pathlib.Path("../../resources/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"spec_auto_{version}.pt")


if __name__ == "__main__":
    main()
