"""Dataset creation routines for polyphonic sample generation."""

# Offline dataset renderer: NumPy + SciPy + SoundFile (+ optional Pedalboard reverb)
# pip install numpy scipy soundfile pedalboard
import csv
import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import soundfile as sf
import torch

from datasets.poly_dataset import PolyphonicAsyncDatasetFromStore
from datasets.poly_utils import (
    RNG,
    Spec,
    log_uniform,
    render_poly_interval_async_freq,
    render_poly_interval_freq,
    render_sample,
)

SR = 44_100
DURATION_S = 1.5  # per sample


def build_dataset_single(
        out_dir="dataset_np",
        n_samples=200,
        freq_range=(100.0, 10_000.0),  # Hz, log-uniform
        sr=SR,
        dur=DURATION_S
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["filename", "f0_hz", "partials", "alpha", "detune_cents", "A", "D", "S", "R",
             "lpf_hz", "drive_db", "reverb_wet", "reverb_room", "sr", "duration_s"])
        for i in range(n_samples):
            f0 = log_uniform(*freq_range)
            spec = Spec()
            y, _ = render_sample(f0, sr, dur, spec)
            fname = f"{i:05d}_f{int(round(f0))}Hz.wav"
            sf.write(out / fname, y, sr, subtype="FLOAT")
            w.writerow([
                fname,
                f0,
                spec.partials,
                spec.alpha,
                spec.detune_cents,
                spec.A1,
                spec.A2,
                spec.C1,
                spec.C2,
                spec.lpf_hz if spec.lpf_hz is not None else "",
                spec.drive_db,
                spec.reverb_wet,
                spec.reverb_room,
                sr,
                dur,
            ])
    # Save a JSON with the RNG seed for reproducibility
    (out / "config.json").write_text(json.dumps(
        {"seed_py": 1234, "seed_np": 1234, "sr": sr, "dur": dur, "freq_range_hz": freq_range}, indent=2
    ))


def build_dataset_poly(
        out_dir="dataset_polyphonic",
        n_samples=200,
        freq_range=(100.0, 10_000.0),  # Hz, log-uniform
        max_polyphony=5,
        sr=SR,
        duration=DURATION_S
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "k_notes", "f0s_hz", "sr", "duration_s"])
        for i in range(n_samples):
            k = RNG.randint(1, max_polyphony)
            freqs = [log_uniform(*freq_range) for _ in range(k)]
            y, f0s = render_poly_interval_freq(freqs, sr, duration)
            fname = f"{i:05d}_k{k}.wav"
            sf.write(out / fname, y, sr, subtype="FLOAT")
            w.writerow([fname, k, ";".join(f"{f0:.6f}" for f0 in f0s), sr, duration])

    (out / "config.json").write_text(json.dumps(
        {"seed_py": 1234, "seed_np": 1234, "sr": sr, "duration": duration,
         "max_polyphony": max_polyphony, "freq_range_hz": freq_range},
        indent=2
    ))


def build_dataset_poly_async(
        out_dir="dataset_polyphonic_async",
        n_samples=200,
        freq_range=(100.0, 10_000.0),  # Hz, log-uniform
        max_polyphony=5,
        sr=SR,
        duration=DURATION_S
):
    """
    Create a dataset of clips. For each clip, pick K~Uniform{1..max_polyphony} base frequencies (log-uniform in freq_range).
    Each note starts/ends at a random time inside the interval.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "k_notes", "f0s_hz", "onsets_s", "durs_s", "sr", "duration_s"
        ])

        for i in range(n_samples):
            k = RNG.randint(1, max_polyphony)
            freqs = [log_uniform(*freq_range) for _ in range(k)]

            y, f0s, onsets_s, durs_s, _ = render_poly_interval_async_freq(freqs, sr, duration)

            fname = f"{i:04d}_k{k}.wav"
            sf.write(out / fname, y, sr, subtype="FLOAT")

            w.writerow([
                fname,
                k,
                ";".join(f"{f0:.6f}" for f0 in f0s),
                ";".join(f"{t:.6f}" for t in onsets_s),
                ";".join(f"{d:.6f}" for d in durs_s),
                sr,
                duration
            ])

    (out / "config.json").write_text(json.dumps(
        {"seed_py": 1234, "seed_np": 1234, "sr": sr, "dur": duration,
         "max_polyphony": max_polyphony, "mode": "async", "freq_range_hz": freq_range},
        indent=2
    ))


def export_store_samples(
        store_path: Union[str, Path],
        destination: Union[str, Path],
        sample_count: int,
        *,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Export random samples from a latent store as audio and label images.

    Args:
        store_path: Directory containing the latent dataset artifacts.
        destination: Output directory where the assets will be written.
        sample_count: Number of random samples to export.
        seed: Optional random seed for reproducible sampling.
        device: Optional torch device used to run the Encodec decoder. Defaults
            to ``"cuda"`` when available otherwise ``"cpu"``.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    root = Path(store_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset store not found: {root}")

    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PolyphonicAsyncDatasetFromStore(root, map_location="cpu")
    dataset_length = len(dataset)
    if sample_count > dataset_length:
        raise ValueError(
            f"Requested {sample_count} samples but dataset only contains {dataset_length}."
        )

    metadata_path = root / "dataset.json"
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dataset metadata JSON: {metadata_path}") from exc

    encoding_meta = metadata.get("encoding", {}) if isinstance(metadata, dict) else {}
    dataset_sr = int(encoding_meta.get("dataset_sample_rate", SR))

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()
    indices = torch.randperm(dataset_length, generator=generator)[:sample_count].tolist()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    from encodec import EncodecModel  # local import to avoid unnecessary dependency at import time
    from encodec.utils import convert_audio

    if dataset_sr == 48_000:
        model = EncodecModel.encodec_model_48khz()
    elif dataset_sr == 24_000:
        model =  EncodecModel.encodec_model_24khz()
    else:
        raise ValueError(f"Dataset sampling rate {dataset_sr} not supported.")
    target_bandwidth = encoding_meta.get("target_bandwidth")
    if target_bandwidth is not None:
        try:
            model.set_target_bandwidth(float(target_bandwidth))
        except (TypeError, ValueError):
            pass
    model = model.to(device)
    model.eval()

    for sample_number, dataset_index in enumerate(indices, start=1):
        latents, label = dataset[dataset_index]

        latents_tensor = latents.detach().to(device=device, dtype=torch.float32)
        with torch.no_grad():
            decoded = model.decoder(latents_tensor.unsqueeze(0))
        waveform = decoded.squeeze(0).to(torch.float32).cpu()

        waveform = convert_audio(
            waveform,
            int(model.sample_rate),
            int(dataset_sr),
            waveform.shape[0],
        )
        audio = waveform.transpose(0, 1).numpy()

        sf.write(output_dir / f"sample{sample_number:03d}.wav", audio, dataset_sr, subtype="FLOAT")

        label_tensor = label.detach().to(torch.float32).cpu()
        label_array = label_tensor.numpy()

        vmax = float(label_array.max())
        plt.imsave(
            output_dir / f"sample_{sample_number:03d}.png",
            label_array,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )


if __name__ == "__main__":
    # build_dataset_poly_async(out_dir="../resources/polyphony_samples", n_samples=50, duration=3, max_polyphony=5)
    export_store_samples(
        store_path="../resources/encodec_latents/poly_async_bandw_test",
        destination="../resources/encodec_latents/samples/bandw_test",
        sample_count=30,
    )
