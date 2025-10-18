"""Dataset creation routines for polyphonic sample generation."""

# Offline dataset renderer: NumPy + SciPy + SoundFile (+ optional Pedalboard reverb)
# pip install numpy scipy soundfile pedalboard
import csv
import json
from pathlib import Path

import soundfile as sf

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


if __name__ == "__main__":
    build_dataset_poly_async(out_dir="../resources/polyphony_samples", n_samples=50, duration=3, max_polyphony=5)
