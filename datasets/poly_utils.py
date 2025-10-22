"""Utility functions and data structures for polyphonic sample generation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, lfilter

try:
    from pedalboard import Pedalboard, Reverb

    HAVE_PEDALBOARD = True
except Exception:
    HAVE_PEDALBOARD = False

RNG = random.Random(1234)
NP_RNG = np.random.default_rng(1234)


def log_uniform(min_hz: float = 100.0, max_hz: float = 10_000.0) -> float:
    """Sample ``f`` ~ LogUniform(min_hz, max_hz) using the NumPy RNG."""
    lo = np.log(float(min_hz))
    hi = np.log(float(max_hz))
    return float(np.exp(NP_RNG.uniform(lo, hi)))


def random_env(n: int, a1: float, a2: float, b: float, c1: float, c2: float) -> np.ndarray:
    knot_vals = np.array([0.0, a1, a2, b, c1, c2, 0.0], dtype=np.float32)
    knot_pos = np.linspace(0.0, n - 1, num=7, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)

    env = np.interp(x, knot_pos, knot_vals).astype(np.float32)
    return env


def biquad_lpf(x: np.ndarray, sr: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    wc = min(max(cutoff_hz / nyq, 1e-5), 0.999)
    b, a = butter(order, wc, btype="low", analog=False)
    return lfilter(b, a, x).astype(np.float32)


def additive_harmonic(
        f0: float,
        sr: float,
        dur: float,
        n_partials: int = 10,
        alpha: float = 1.0,
        detune_cents_std: float = 3.0,
        phase_random: bool = True,
) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    partial_idxs = np.arange(1, n_partials + 1, dtype=np.float32)
    amps = (1.0 / (partial_idxs ** alpha)).astype(np.float32)
    amps *= NP_RNG.uniform(0.7, 1., size=n_partials).astype(np.float32)
    detune = NP_RNG.normal(0.0, detune_cents_std, size=n_partials).astype(np.float32)
    freqs = f0 * partial_idxs * (2.0 ** (detune / 1200.0))
    phases = (
        NP_RNG.uniform(0, 2 * np.pi, size=n_partials).astype(np.float32)
        if phase_random
        else np.zeros(n_partials, np.float32)
    )
    y = np.zeros(n, dtype=np.float32)
    for a, f, p in zip(amps, freqs, phases):
        y += a * np.sin(2 * np.pi * f * t + p, dtype=np.float32)
    return y


def simple_clip(x: np.ndarray, drive: float = 0.0) -> np.ndarray:
    g = 10.0 ** (drive / 20.0)
    return np.tanh(g * x).astype(np.float32)


def apply_reverb_offline(x: np.ndarray, sr: float, wet: float = 0.15, room_size: float = 0.3) -> np.ndarray:
    if not HAVE_PEDALBOARD:
        return x
    board = Pedalboard([Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet)])
    return board(x.reshape(1, -1), sample_rate=sr).reshape(-1).astype(np.float32)


def _optional_uniform(low: float, high: float) -> Optional[float]:
    return RNG.choice([None, RNG.uniform(low, high)])


def _optional_uniform_with_zero(low: float, high: float) -> float:
    return RNG.choice([0.0, RNG.uniform(low, high)])


@dataclass
class Spec:
    """Random synthesis specification mimicking the previous ``random_spec`` output."""

    partials: int = field(default_factory=lambda: RNG.randint(5, 24))
    alpha: float = field(default_factory=lambda: RNG.uniform(0.6, 2.4))
    detune_cents: float = field(default_factory=lambda: RNG.uniform(0.0, 8.0))
    A1: float = field(default_factory=lambda: RNG.uniform(0.0, 1.0))
    A2: float = field(default_factory=lambda: RNG.uniform(0.0, 1.0))
    B: float = field(default_factory=lambda: RNG.uniform(0.1, 1.0))
    C1: float = field(default_factory=lambda: RNG.uniform(0.0, 1.0))
    C2: float = field(default_factory=lambda: RNG.uniform(0.0, 1.0))
    lpf_hz: Optional[float] = field(default_factory=lambda: _optional_uniform(1_500, 12_000))
    drive_db: float = field(default_factory=lambda: _optional_uniform_with_zero(1.5, 10.0))
    reverb_wet: float = field(default_factory=lambda: _optional_uniform_with_zero(0.05, 0.25))
    reverb_room: float = field(default_factory=lambda: RNG.uniform(0.1, 0.6))


def render_sample(
        f0: float,
        sr: float,
        dur: float,
        spec: Spec,
) -> np.ndarray:
    n = int(dur * sr)
    tone = additive_harmonic(
        f0=f0,
        sr=sr,
        dur=dur,
        n_partials=spec.partials,
        alpha=spec.alpha,
        detune_cents_std=spec.detune_cents,
    )
    env = random_env(n, spec.A1, spec.A2, spec.B, spec.C1, spec.C2)
    y = (tone * env).astype(np.float32)
    if spec.lpf_hz is not None:
        y = biquad_lpf(y, sr, cutoff_hz=spec.lpf_hz, order=4)
    if spec.drive_db != 0.0:
        y = simple_clip(y, drive=spec.drive_db)
    if spec.reverb_wet > 0.0:
        y = apply_reverb_offline(y, sr, wet=spec.reverb_wet, room_size=spec.reverb_room)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y


def render_poly_interval_freq(freqs_hz: Sequence[float], sr: float, dur: float) -> Tuple[np.ndarray, List[float]]:
    mix = np.zeros(int(dur * sr), dtype=np.float32)
    f0s: List[float] = []
    for f0 in freqs_hz:
        f0_float = float(f0)
        f0s.append(f0_float)
        spec = Spec()
        note = render_sample(f0_float, sr, dur, spec)
        mix += note
    peak = np.max(np.abs(mix)) + 1e-12
    mix = (0.95 * mix / peak).astype(np.float32)
    return mix, f0s


def render_poly_interval_async_freq(
        freqs_hz: Sequence[float],
        sr: float,
        dur: float,
        min_note_dur: float = 0.12,
) -> Tuple[np.ndarray, List[float], List[float], List[float], List[np.ndarray]]:
    n_total = int(dur * sr)
    mix = np.zeros(n_total, dtype=np.float32)

    f0s: List[float] = []
    onsets_s: List[float] = []
    durs_s: List[float] = []
    samples: List[np.ndarray] = []

    for f0 in freqs_hz:
        f0_float = float(f0)
        f0s.append(f0_float)

        onset_s = RNG.uniform(0.0, dur - min_note_dur)
        max_dur = max(min(dur - onset_s, dur), min_note_dur)
        note_dur = RNG.uniform(min_note_dur, max_dur)

        spec = Spec()
        y = render_sample(f0_float, sr, note_dur, spec)

        start = int(onset_s * sr)
        end = min(start + len(y), n_total)

        y_embedded = np.zeros(n_total, dtype=np.float32)
        y_embedded[start:end] += y[: end - start] ** 2
        samples.append(y_embedded)

        mix += y_embedded

        onsets_s.append(onset_s)
        durs_s.append(note_dur)

    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak
        samples = [sample / peak for sample in samples]
    return mix, f0s, onsets_s, durs_s, samples


def power(x, sr, fc=12.0, out_sr=75.0, method="filtfilt"):
    """ Broadband power envelope from an audio signal. """
    if fc >= out_sr / 2:
        fc = 0.9 * (out_sr / 2)  # safety

    # Instantaneous power
    p = x.astype(float) ** 2

    # Low-pass on power
    if method == "iir":
        # One-pole (exponential moving average), fc -> alpha
        alpha = 1.0 - np.exp(-2.0 * np.pi * fc / sr)
        y = np.empty_like(p)
        y[0] = p[0]
        for n in range(1, len(p)):
            y[n] = y[n - 1] + alpha * (p[n] - y[n - 1])
    elif method == "filtfilt":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, fc / (sr / 2.0))  # 4th-order Butterworth
        y = filtfilt(b, a, p, padlen=min(3 * max(len(b), len(a)), len(p) - 1))
    else:
        raise ValueError("method must be 'iir' or 'filtfilt'")

    # Resample to 75 Hz by interpolation (aligns to exact 1/75 s grid)
    t = np.arange(len(y)) / sr
    t_75 = np.arange(0.0, t[-1] + 1e-12, 1.0 / out_sr)
    pow_75 = np.interp(t_75, t, y)

    return pow_75
