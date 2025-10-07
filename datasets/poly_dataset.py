"""PyTorch dataset for generating asynchronous polyphonic audio clips."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from datasets import poly_utils


Label = List[Tuple[float, float, float]]


def _validate_freq_range(freq_range: Sequence[float]) -> Tuple[float, float]:
    if not isinstance(freq_range, Iterable):
        raise TypeError("freq_range must be an iterable with two elements")
    freq_tuple = tuple(float(f) for f in freq_range)
    if len(freq_tuple) != 2:
        raise ValueError("freq_range must contain exactly two values")
    min_freq, max_freq = freq_tuple
    if not math.isfinite(min_freq) or not math.isfinite(max_freq):
        raise ValueError("freq_range values must be finite")
    if min_freq <= 0.0 or max_freq <= 0.0:
        raise ValueError("freq_range values must be positive")
    if min_freq > max_freq:
        raise ValueError("freq_range minimum must be <= maximum")
    return min_freq, max_freq


class PolyphonicAsyncDataset(Dataset[Tuple[Tensor, Label]]):
    """Dataset that renders asynchronous polyphonic mixtures on the fly."""

    def __init__(
        self,
        *,
        n_samples: int,
        freq_range: Sequence[float],
        max_polyphony: int,
        sr: int,
        duration: float,
        min_note_duration: float = 0.12,
        seed: Optional[int] = None,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if max_polyphony <= 0:
            raise ValueError("max_polyphony must be positive")
        if sr <= 0:
            raise ValueError("sr must be positive")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if min_note_duration <= 0.0:
            raise ValueError("min_note_duration must be positive")

        self.n_samples = int(n_samples)
        self.max_polyphony = int(max_polyphony)
        self.sample_rate = int(sr)
        self.duration = float(duration)
        self.min_note_duration = float(min_note_duration)
        self.freq_min, self.freq_max = _validate_freq_range(freq_range)

        self._base_seed = seed if seed is not None else 1234
        self._sample_seeds = [self._base_seed + i for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Label]:
        if index < 0 or index >= self.n_samples:
            raise IndexError("index out of range")

        seed = self._sample_seeds[index]
        waveform, f0s, onsets, durs = self._render_sample(seed)

        audio = torch.from_numpy(waveform).to(dtype=torch.float32)
        audio = audio.unsqueeze(0)

        labels: Label = [
            (float(freq), float(onset), float(onset + dur))
            for freq, onset, dur in zip(f0s, onsets, durs)
        ]

        return audio, labels

    def _render_sample(
        self, seed: int
    ) -> Tuple[np.ndarray, Sequence[float], Sequence[float], Sequence[float]]:
        rng_state = poly_utils.RNG.getstate()
        np_rng = poly_utils.NP_RNG
        try:
            poly_utils.RNG.seed(seed)
            poly_utils.NP_RNG = np.random.default_rng(seed)

            k = poly_utils.RNG.randint(1, self.max_polyphony)
            freqs = [
                poly_utils.log_uniform(self.freq_min, self.freq_max)
                for _ in range(k)
            ]
            waveform, f0s, onsets, durs = poly_utils.render_poly_interval_async_freq(
                freqs,
                self.sample_rate,
                self.duration,
                min_note_dur=self.min_note_duration,
            )
        finally:
            poly_utils.RNG.setstate(rng_state)
            poly_utils.NP_RNG = np_rng

        return waveform, f0s, onsets, durs
