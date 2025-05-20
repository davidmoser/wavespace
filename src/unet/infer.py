import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import torch
import torchaudio

from unet_arch_v1 import AudioUNet  # your model file


def crop_to_multiple(x: torch.Tensor, m: int = 8, dims=(-2, -1)):
    sl = [slice(None)] * x.ndim
    for d in dims:
        sl[d] = slice(0, x.shape[d] - (x.shape[d] % m))
    return x[tuple(sl)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_wav", type=pathlib.Path, required=True,
                   help="input audio file (wav/mp3)")
    p.add_argument("--ckpt", type=pathlib.Path, default="../../resources/checkpoints/audio_unet_v1.pt")
    p.add_argument("--out_img", type=pathlib.Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    version = re.search(r"_(v\d\d?)\.pt$", args.ckpt.name).group(1)
    sr, n_fft, hop, n_mels = 22_050, 4096, 512, 256
    # load & trim audio
    wav, orig_sr = torchaudio.load(args.in_wav)
    wav = wav.mean(0, keepdim=True)  # mono
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    wav = wav[..., : int(20 * sr)]  # first 20 s

    # analysis transform
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    with torch.no_grad():  # no grad for STFT
        mel = to_db(spec(wav))  # (1, F, T)
    mel = crop_to_multiple(mel, 16)  # F=256 already OK
    x = mel.unsqueeze(0).to(args.device)  # (1,1,F,T)

    # load model
    model = AudioUNet().to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    # inference
    with torch.no_grad():
        y = model(x).cpu()[0, 0]  # (F,T)

    # plot & save
    fig, axes = plt.subplots(2, 1, figsize=(y.shape[1] / 100, y.shape[0] / 50),
                             sharex=True)
    for ax, data, title in zip(axes, [mel[0], y], ["Original", "Reconstruction"]):
        im = ax.imshow(data, origin="lower", aspect="auto",
                       cmap="coolwarm", interpolation="none")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout(pad=0)
    out_img = args.out_img if args.out_img else args.in_wav.with_name(f"{args.in_wav.stem}_{version}.png")
    plt.savefig(out_img, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"saved {out_img}")

    # save as wav
    out_wav = args.in_wav.with_name(f"{args.in_wav.stem}_{version}.wav")

    # 1. dB → power  (we used power-spectrograms: power=2)
    mel_power = torchaudio.functional.DB_to_amplitude(y, ref=1.0, power=2.0)

    # 2. mel → linear-frequency power
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sr).to(y.device)
    S_power = inv_mel(mel_power)

    # 3. power → magnitude, then Griffin–Lim (phase estimate)
    S_mag = S_power.sqrt()  # magnitude
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=32).to(y.device)
    wav_hat = griffin(S_mag)  # (T,)

    torchaudio.save(out_wav, wav_hat.unsqueeze(0), sr,
                    encoding="PCM_S", bits_per_sample=16)
    print(f"saved {out_wav}")


if __name__ == "__main__":
    main()
