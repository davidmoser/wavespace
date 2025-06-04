import json
import pathlib
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.audio import AudioFile
from demucs.pretrained import get_model

from spectrogram_converter.convert import calculate_log_matrix


def wav_to_spec(
        path: str,
        sr: int = 24_000,
        n_fft: int = 1024,
        hop_ratio: float = 0.25,
        mono: bool = True):
    wav, orig_sr = torchaudio.load(path)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=int(n_fft * hop_ratio),
                      window=torch.hann_window(n_fft), return_complex=True)
    mag = stft.abs()  # (1, F, T)
    return mag


def preprocess(example, sr=24_000):
    # example["audio"]["array"] is 1‑D float32 @ native SR
    wav = torch.tensor(example["audio"]["array"]).unsqueeze(0)
    mag = wav_to_spec_from_tensor(wav, example["audio"]["sampling_rate"], sr)
    return {"mag": mag}


def wav_to_spec_from_tensor(wav, orig_sr, target_sr,
                            n_fft=1024, hop_ratio=0.25):
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=int(n_fft * hop_ratio),
                      window=torch.hann_window(n_fft), return_complex=True)
    return stft.abs()


def wav_to_spectrogram_image(
        in_wav: str,
        out_img: str = "spectrogram.png",
        sr: int = 24_000,
        n_fft: int = 1024,
        hop: int = 128,
        cmap: str = "coolwarm"  # blue→red gradient
) -> None:
    # load & resample ---------------------------------------------------------
    wav, orig_sr = torchaudio.load(in_wav, num_frames=1_000_000)
    wav = wav.mean(0, keepdim=True)  # mono
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    # magnitude spectrogram ---------------------------------------------------
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop,
        window=torch.hann_window(n_fft), return_complex=True
    ).abs()

    # log power (dB) for better visual contrast -------------------------------
    spec_db = 20 * torch.log10(spec + 1e-9)
    h, w = spec_db[0].shape

    # plot & save -------------------------------------------------------------
    plt.figure(figsize=(w / 100, h / 100))
    plt.imshow(spec_db[0], origin="lower", aspect="auto", cmap=cmap, interpolation="none")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_img, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def wav_to_mel_spectrogram_image(
        in_wav: str,
        out_img: str = "mel_spectrogram.png",
        sr: int = 24_000,
        n_fft: int = 1024,
        hop: int = 128,
        n_mels: int = 1024 // 8,
        cmap: str = "coolwarm"  # try magma
) -> None:
    """Convert a WAV/MP3 file to a log-magnitude mel-spectrogram PNG."""
    # load & resample ---------------------------------------------------------
    wav, orig_sr = torchaudio.load(in_wav, num_frames=1_000_000)
    wav = wav.mean(0, keepdim=True)  # mono
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    # mel spectrogram ---------------------------------------------------------
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0  # magnitude² → power
    )(wav)
    spec_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    h, w = spec_db.shape[-2:]

    # plot & save -------------------------------------------------------------
    plt.figure(figsize=(w / 100, h / 100))
    plt.imshow(spec_db[0], origin="lower", aspect="auto",
               cmap=cmap, interpolation="none")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_img, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def wav_to_log_spectrogram_image(
        in_wav: str,
        out_img: str = "log_spectrogram.png",
        sr: int = 24_000,
        n_fft: int = 4096,
        hop: int = 512,
        log_bins: int = 256,
        cmap: str = "coolwarm"  # try magma
) -> None:
    """Convert a WAV/MP3 file to a log-magnitude mel-spectrogram PNG."""
    # load & resample ---------------------------------------------------------
    wav, orig_sr = torchaudio.load(in_wav, num_frames=1_000_000)
    wav = wav.mean(0, keepdim=True)  # mono
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    # log spectrogram ---------------------------------------------------------
    spec_fft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop, power=2.0)(wav)  # (C=1, F, T)
    W = calculate_log_matrix(n_fft, sr, log_bins)

    spec_log = torch.matmul(W, spec_fft[0]).unsqueeze(0)

    spec_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec_log)
    h, w = spec_db.shape[-2:]

    # plot & save -------------------------------------------------------------
    plt.figure(figsize=(w / 100, h / 100))
    plt.imshow(spec_db[0], origin="lower", aspect="auto",
               cmap=cmap, interpolation="none")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_img, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def extract_tracks(path: str):
    src = pathlib.Path(path)  # e.g. python split_tracks.py input.mp4
    probe = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "a",  # audio streams only
        "-show_entries", "stream=index",
        "-of", "json", str(src)])
    for s in json.loads(probe)["streams"]:
        i = s["index"]  # stream index inside the file
        out = src.with_suffix(f".track{i}.m4a")
        subprocess.check_call([
            "ffmpeg", "-y",  # overwrite if exists
            "-i", str(src),
            "-map", f"0:a:{i}",  # pick one audio stream
            "-c", "copy",  # no re-encoding
            str(out)])
        print("wrote", out)


def run_demucs(audio_path: str, out_dir: str, model_name: str = "htdemucs", device: str = None):
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device).eval()

    wav = AudioFile(audio_path).read(streams=[0],
                                     samplerate=model.samplerate,
                                     channels=model.audio_channels)

    mix = wav.to(device)
    sources = apply_model(model, mix, device=device)[0]

    target_dir = out_dir / model_name / audio_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    for tensor, name in zip(sources, model.sources):
        torchaudio.save(str(target_dir / f"{name}.wav"), tensor.cpu(), model.samplerate)
        print(f"✓ {target_dir / name}.wav")
