"""Utilities for generating deterministic sine wave datasets."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SineDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset that yields mono sine waves with randomised frequencies."""

    def __init__(
        self,
        duration: float,
        sampling_rate: int,
        min_frequency: float,
        max_frequency: float,
        num_samples: int,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            duration: Length of each audio clip in seconds.
            sampling_rate: Sampling rate used for the generated audio.
            min_frequency: Minimum frequency (Hz) sampled for the sine waves.
            max_frequency: Maximum frequency (Hz) sampled for the sine waves.
            num_samples: Number of audio samples in the dataset.
            seed: Optional random seed to make the dataset deterministic.
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        if sampling_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if min_frequency <= 0 or max_frequency <= 0:
            raise ValueError("Frequencies must be positive")
        if min_frequency > max_frequency:
            raise ValueError("min_frequency cannot be greater than max_frequency")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.duration = float(duration)
        self.sample_rate = int(sampling_rate)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.num_samples = int(num_samples)

        self._num_audio_samples = int(round(self.duration * self.sample_rate))
        if self._num_audio_samples <= 0:
            raise ValueError("duration and sample_rate combination produces no samples")

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self._frequencies = torch.empty(self.num_samples)
        self._phases = torch.empty(self.num_samples)
        self._frequencies.uniform_(self.min_frequency, self.max_frequency, generator=generator)
        self._phases.uniform_(0.0, 2.0 * math.pi, generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        frequency = self._frequencies[index]
        phase = self._phases[index]

        time = torch.arange(self._num_audio_samples, dtype=torch.float32) / self.sample_rate
        waveform = torch.sin(2.0 * math.pi * frequency * time + phase)

        # Ensure output is mono (single channel)
        waveform = waveform.unsqueeze(0)

        label = frequency.to(dtype=torch.float32)
        return waveform, label
