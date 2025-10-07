# Offline dataset renderer: NumPy + SciPy + SoundFile (+ optional Pedalboard reverb)
# pip install numpy scipy soundfile pedalboard
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

try:
    from pedalboard import Pedalboard, Reverb

    HAVE_PEDALBOARD = True
except Exception:
    HAVE_PEDALBOARD = False

SR = 44_100
DURATION_S = 1.5  # per sample
RNG = random.Random(1234)
NP_RNG = np.random.default_rng(1234)


# --- helpers ---
def log_uniform(min_hz=100.0, max_hz=10_000.0):
    """Sample f ~ LogUniform(min_hz, max_hz) using the NumPy RNG for reproducibility."""
    lo = np.log(float(min_hz))
    hi = np.log(float(max_hz))
    return float(np.exp(NP_RNG.uniform(lo, hi)))


def adsr_env(n, sr, a=0.01, d=0.05, s=0.6, r=0.2):
    a_n = int(a * sr)
    d_n = int(d * sr)
    r_n = int(r * sr)
    s_n = max(0, n - (a_n + d_n + r_n))
    if a_n < 1: a_n = 1
    if d_n < 1: d_n = 1
    if r_n < 1: r_n = 1
    env = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s, dtype=np.float32),
        np.linspace(s, 0, r_n, endpoint=True)
    ]).astype(np.float32)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    else:
        env = env[:n]
    return env


def biquad_lpf(x, sr, cutoff_hz, order=4):
    # Butterworth low-pass (stable, simple)
    nyq = 0.5 * sr
    wc = min(max(cutoff_hz / nyq, 1e-5), 0.999)
    b, a = butter(order, wc, btype='low', analog=False)
    return lfilter(b, a, x).astype(np.float32)


def additive_harmonic(f0, sr, dur, n_partials=10, alpha=1.0, detune_cents_std=3.0, phase_random=True):
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # 1/n^alpha spectral tilt
    partial_idxs = np.arange(1, n_partials + 1, dtype=np.float32)
    amps = (1.0 / (partial_idxs ** alpha)).astype(np.float32)
    amps /= amps.max()
    # small randomization per partial
    amps *= NP_RNG.uniform(0.85, 1.15, size=n_partials).astype(np.float32)
    # detune in cents
    detune = NP_RNG.normal(0.0, detune_cents_std, size=n_partials).astype(np.float32)
    freqs = f0 * partial_idxs * (2.0 ** (detune / 1200.0))
    phases = NP_RNG.uniform(0, 2 * np.pi, size=n_partials).astype(np.float32) if phase_random else np.zeros(n_partials,
                                                                                                            np.float32)
    # accumulate
    y = np.zeros(n, dtype=np.float32)
    for a, f, p in zip(amps, freqs, phases):
        y += a * np.sin(2 * np.pi * f * t + p, dtype=np.float32)
    # normalize to peak 1 before envelope/effects
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


def simple_clip(x, drive=0.0):
    # soft saturation via tanh; drive in dB
    g = 10.0 ** (drive / 20.0)
    return np.tanh(g * x).astype(np.float32)


def apply_reverb_offline(x, sr, wet=0.15, room_size=0.3):
    if not HAVE_PEDALBOARD:
        return x
    board = Pedalboard([Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet)])
    return board(x.reshape(1, -1), sample_rate=sr).reshape(-1).astype(np.float32)


# --- dataset generation ---
def render_sample(f0, sr, dur, spec):
    n = int(dur * sr)
    # Synthesis
    tone = additive_harmonic(
        f0=f0,
        sr=sr,
        dur=dur,
        n_partials=spec["partials"],
        alpha=spec["alpha"],
        detune_cents_std=spec["detune_cents"]
    )
    # Envelope
    env = adsr_env(n, sr, a=spec["A"], d=spec["D"], s=spec["S"], r=spec["R"])
    y = (tone * env).astype(np.float32)
    # Optional filtering
    if spec["lpf_hz"] is not None:
        y = biquad_lpf(y, sr, cutoff_hz=spec["lpf_hz"], order=4)
    # Optional drive
    if spec["drive_db"] != 0.0:
        y = simple_clip(y, drive=spec["drive_db"])
    # Optional reverb
    if spec["reverb_wet"] > 0.0:
        y = apply_reverb_offline(y, sr, wet=spec["reverb_wet"], room_size=spec["reverb_room"])
    # Final normalize to -1..1 with a little headroom
    peak = np.max(np.abs(y)) + 1e-12
    y = (0.95 * y / peak).astype(np.float32)
    return y


def random_spec():
    return {
        "partials": RNG.randint(5, 24),
        "alpha": RNG.uniform(0.6, 2.4),
        "detune_cents": RNG.uniform(0.0, 8.0),
        "A": RNG.uniform(0.005, 0.06),
        "D": RNG.uniform(0.03, 0.2),
        "S": RNG.uniform(0.4, 0.95),
        "R": RNG.uniform(0.08, 0.35),
        "lpf_hz": RNG.choice([None, RNG.uniform(1_500, 12_000)]),
        "drive_db": RNG.choice([0.0, RNG.uniform(1.5, 10.0)]),
        "reverb_wet": RNG.choice([0.0, RNG.uniform(0.05, 0.25)]),
        "reverb_room": RNG.uniform(0.1, 0.6),
    }


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
            spec = random_spec()
            y = render_sample(f0, sr, dur, spec)
            fname = f"{i:05d}_f{int(round(f0))}Hz.wav"
            sf.write(out / fname, y, sr, subtype="FLOAT")
            w.writerow([
                fname, f0,
                spec["partials"], spec["alpha"], spec["detune_cents"],
                spec["A"], spec["D"], spec["S"], spec["R"],
                spec["lpf_hz"] if spec["lpf_hz"] is not None else "",
                spec["drive_db"], spec["reverb_wet"], spec["reverb_room"],
                sr, dur
            ])
    # Save a JSON with the RNG seed for reproducibility
    (out / "config.json").write_text(json.dumps(
        {"seed_py": 1234, "seed_np": 1234, "sr": sr, "dur": dur, "freq_range_hz": freq_range}, indent=2
    ))


def render_poly_interval_freq(freqs_hz, sr, dur):
    """Render an interval containing all given base frequencies played simultaneously."""
    mix = np.zeros(int(dur * sr), dtype=np.float32)
    f0s = []
    for f0 in freqs_hz:
        f0s.append(float(f0))
        spec = random_spec()
        note = render_sample(float(f0), sr, dur, spec)
        mix += note
    # normalize poly mix with headroom
    peak = np.max(np.abs(mix)) + 1e-12
    mix = (0.95 * mix / peak).astype(np.float32)
    return mix, f0s


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


# --- asynchronous (staggered) polyphonic interval renderer ---
def render_poly_interval_async_freq(freqs_hz, sr, dur, min_note_dur=0.12):
    """
    Render an interval where each base frequency has a random onset and duration within the interval.
    Notes may overlap arbitrarily.
    """
    n_total = int(dur * sr)
    mix = np.zeros(n_total, dtype=np.float32)

    f0s, onsets_s, durs_s = [], [], []
    for f0 in freqs_hz:
        f0 = float(f0)
        f0s.append(f0)

        # random onset and duration inside [0, dur]
        onset_s = RNG.uniform(0.0, dur - min_note_dur)
        max_dur = max(min(dur - onset_s, dur), min_note_dur)
        note_dur = RNG.uniform(min_note_dur, max_dur)

        # synthesize this note with its own random spec and duration
        spec = random_spec()
        y = render_sample(f0, sr, note_dur, spec)

        # mix in at onset
        start = int(onset_s * sr)
        end = min(start + len(y), n_total)
        mix[start:end] += y[: end - start]

        onsets_s.append(onset_s)
        durs_s.append(note_dur)

    # normalize poly mix with headroom
    peak = float(np.max(np.abs(mix)) + 1e-12)
    mix = (0.95 * mix / peak).astype(np.float32)
    return mix, f0s, onsets_s, durs_s


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

            y, f0s, onsets_s, durs_s = render_poly_interval_async_freq(freqs, sr, duration)

            fname = f"{i:05d}_k{k}_async.wav"
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
