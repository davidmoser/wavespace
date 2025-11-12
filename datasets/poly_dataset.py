"""PyTorch dataset for generating asynchronous polyphonic audio clips."""

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from datasets import poly_utils
from datasets.poly_utils import power
from pitch_detection_supervised.utils import events_to_active_label


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


class PolyphonicAsyncDataset(Dataset[Tuple[Tensor, Tensor]]):
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
            label_sample_rate: float = 75.0,
            label_centers_hz: Optional[Sequence[float]] = None,
            label_bins: int = 128,
            label_type: str = "power",
            seed: Optional[int] = None,
    ) -> None:
        self.n_samples = int(n_samples)
        self.max_polyphony = int(max_polyphony)
        self.sample_rate = int(sr)
        self.duration = float(duration)
        self.min_note_duration = float(min_note_duration)
        self.freq_min, self.freq_max = _validate_freq_range(freq_range)

        self.label_sample_rate = float(label_sample_rate)
        self.label_frames = max(1, int(round(self.duration * self.label_sample_rate)))

        if label_centers_hz is None:
            centers = np.geomspace(self.freq_min, self.freq_max, num=int(label_bins)).astype(np.float32)
        else:
            centers = np.asarray(list(label_centers_hz), dtype=np.float32)
            if centers.ndim != 1 or centers.size == 0:
                raise ValueError("label_centers_hz must be a non-empty 1D sequence")
            centers = np.sort(centers)
        self._label_centers = centers
        self._label_bins = int(centers.shape[0])

        label_type_value = str(label_type).lower()
        if label_type_value not in {"power", "activation"}:
            raise ValueError("label_type must be either 'power' or 'activation'")

        self.label_type = label_type_value

        self._base_seed = seed if seed is not None else 1234
        self._sample_seeds = [self._base_seed + i for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index < 0 or index >= self.n_samples:
            raise IndexError("index out of range")

        seed = self._sample_seeds[index]
        mix, f0s, onsets, durations, samples = self._render_sample(seed)

        audio = torch.from_numpy(mix).to(dtype=torch.float32)
        audio = audio.unsqueeze(0)

        if self.label_type == "power":
            label = self._build_power_label(f0s, samples)
        elif self.label_type == "activation":
            label = self._build_activation_label(f0s, onsets, durations)
        else:
            raise RuntimeError(f"Unsupported label_type: {self.label_type}")

        return audio, label

    def _render_sample(
            self, seed: int
    ) -> Tuple[np.ndarray, List[float], List[float], List[float], List[np.ndarray]]:
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
            mix, f0s, onsets, durations, samples = poly_utils.render_poly_interval_async_freq(
                freqs,
                self.sample_rate,
                self.duration,
                min_note_dur=self.min_note_duration,
            )
        finally:
            poly_utils.RNG.setstate(rng_state)
            poly_utils.NP_RNG = np_rng

        return mix, f0s, onsets, durations, samples

    def _build_power_label(
            self, f0s: List[float], samples: Sequence[np.ndarray]
    ) -> Tensor:
        label = torch.zeros(self._label_bins, self.label_frames, dtype=torch.float32)

        for f0, sample in zip(f0s, samples):
            weights = self._frequency_weights(f0)
            if not weights:
                continue

            pow_y = power(sample, self.sample_rate, method="filtfilt", out_sr=self.label_sample_rate)

            for bin_index, weight in weights:
                label[bin_index] += pow_y * weight

        label.clamp_(max=1.0)

        return label

    def _build_activation_label(
            self, f0s: List[float], onsets: List[float], durations: List[float]
    ) -> Tensor:
        offsets = [onset + duration for onset, duration in zip(onsets, durations)]
        events = zip(f0s, onsets, offsets)

        if not events:
            return torch.zeros(self._label_bins, self.label_frames, dtype=torch.float32)

        label = events_to_active_label(
            events,
            self._label_centers,
            self.duration,
            self.label_frames,
            dtype=torch.float32,
        )

        return label

    def _frequency_weights(self, freq_hz: float) -> List[Tuple[int, float]]:
        centers = self._label_centers
        if freq_hz <= centers[0]:
            return [(0, 1.0)]
        if freq_hz >= centers[-1]:
            return [(self._label_bins - 1, 1.0)]

        idx = int(np.searchsorted(centers, freq_hz, side="left"))
        if idx == 0:
            return [(0, 1.0)]
        if idx >= self._label_bins:
            return [(self._label_bins - 1, 1.0)]

        left = centers[idx - 1]
        right = centers[idx]
        denom = float(max(right - left, np.finfo(np.float32).eps))
        alpha = float((freq_hz - left) / denom)
        alpha = min(max(alpha, 0.0), 1.0)
        return [(idx - 1, 1.0 - alpha), (idx, alpha)]

    def get_sample_rate(self) -> int: return self.sample_rate