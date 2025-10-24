"""PyTorch dataset for generating asynchronous polyphonic audio clips."""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import zstandard
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

        if label_sample_rate <= 0.0:
            raise ValueError("label_sample_rate must be positive")
        if label_bins <= 0:
            raise ValueError("label_bins must be positive")

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
        mix, f0s, samples = self._render_sample(seed)

        audio = torch.from_numpy(mix).to(dtype=torch.float32)
        audio = audio.unsqueeze(0)

        label = self._build_label_tensor(f0s, samples)

        return audio, label

    def _render_sample(
            self, seed: int
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
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
            mix, f0s, _, _, samples = poly_utils.render_poly_interval_async_freq(
                freqs,
                self.sample_rate,
                self.duration,
                min_note_dur=self.min_note_duration,
            )
        finally:
            poly_utils.RNG.setstate(rng_state)
            poly_utils.NP_RNG = np_rng

        return mix, f0s, samples

    def _build_label_tensor(
            self, f0s: List[float], samples: Sequence[np.ndarray]
    ) -> Tensor:
        if self.label_type == "power":
            return self._build_power_label(f0s, samples)
        if self.label_type == "activation":
            return self._build_activation_label(f0s, samples)
        raise RuntimeError(f"Unsupported label_type: {self.label_type}")

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
            self, f0s: List[float], samples: Sequence[np.ndarray]
    ) -> Tensor:
        events = []
        sr = float(self.sample_rate)
        threshold = 0.1
        for f0, sample in zip(f0s, samples):
            if sample is None:
                continue
            sample_arr = np.asarray(sample, dtype=np.float32)
            if sample_arr.size == 0:
                continue

            amplitude = np.sqrt(np.clip(sample_arr, 0.0, None))
            active = amplitude > threshold
            if not np.any(active):
                continue

            active_indices = np.flatnonzero(active)
            splits = np.split(active_indices, np.where(np.diff(active_indices) > 1)[0] + 1)
            for segment in splits:
                if segment.size == 0:
                    continue
                start_idx = int(segment[0])
                end_idx = int(segment[-1]) + 1
                onset = float(start_idx / sr)
                offset = float(end_idx / sr)
                if offset <= onset:
                    continue
                if onset >= self.duration:
                    continue
                events.append((float(f0), max(0.0, onset), min(self.duration, offset)))

        if not events:
            return torch.zeros(self._label_bins, self.label_frames, dtype=torch.float32)

        label = events_to_active_label(
            events,
            self._label_centers,
            self.duration,
            self.label_frames,
            dtype=torch.float32,
        )

        return label.gt(0).to(dtype=torch.float32)

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


class PolyphonicAsyncDatasetFromStore(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset backed by artifacts produced with :func:`create_latent_store`.

    The dataset eagerly loads the manifest labels and index information into
    memory. Samples are retrieved by seeking into the compressed shard using the
    cached offsets, allowing efficient random access without scanning the whole
    archive.
    """

    def __init__(
            self,
            store_path: Union[str, Path],
            *,
            map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> None:
        self._root = Path(store_path)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Dataset store not found: {self._root}")

        self._map_location = map_location
        self._decompressor = zstandard.ZstdDecompressor()

        index_path = self._root / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing index file: {index_path}")

        self._records = self._load_index(index_path)
        if not self._records:
            raise ValueError("The dataset index is empty.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self._records):
            raise IndexError("index out of range")

        record = self._records[index]
        latents, label = self._load_payload(record)

        return latents, label

    class _IndexRecord:
        __slots__ = ("key", "shard", "member", "offset", "size")

        def __init__(
                self,
                key: str,
                shard: str,
                member: str,
                offset: int,
                size: int,
        ) -> None:
            self.key = key
            self.shard = shard
            self.member = member
            self.offset = offset
            self.size = size

    def _load_index(self, index_path: Path) -> List["PolyphonicAsyncDatasetFromStore._IndexRecord"]:
        records: List[PolyphonicAsyncDatasetFromStore._IndexRecord] = []
        with index_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "__meta__" in data:
                    continue

                try:
                    key = str(data["key"])
                    shard = str(data["shard"])
                    member = str(data["member"])
                    offset = int(data["offset"])
                    size = int(data["size"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid record in index.jsonl at line {line_number + 1}: {data}"
                    ) from exc

                records.append(self._IndexRecord(key, shard, member, offset, size))

        return records

    def _load_payload(self, record: "PolyphonicAsyncDatasetFromStore._IndexRecord") -> Tuple[Tensor, Tensor]:
        shard_path = self._root / record.shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        with shard_path.open("rb") as file:
            file.seek(record.offset)
            compressed = file.read(record.size)

        if not compressed:
            raise ValueError(
                f"Empty compressed payload for key '{record.key}' in shard '{record.shard}'."
            )

        with self._decompressor.stream_reader(io.BytesIO(compressed)) as reader:
            decompressed = reader.read()

        buffer = io.BytesIO(decompressed)

        header = buffer.read(tarfile.BLOCKSIZE)
        if len(header) != tarfile.BLOCKSIZE:
            raise ValueError(
                f"Invalid tar member for key '{record.key}' in shard '{record.shard}'."
            )

        tar_info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="surrogateescape")
        if tar_info.name != record.member:
            raise ValueError(
                "Tar member name mismatch: "
                f"expected '{record.member}' but found '{tar_info.name}'."
            )
        data_size = tar_info.size
        payload = buffer.read(((data_size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE)
        if len(payload) < data_size:
            raise ValueError(
                f"Corrupted payload for key '{record.key}' in shard '{record.shard}'."
            )

        file_data = payload[:data_size]
        payload = torch.load(io.BytesIO(file_data), map_location=self._map_location)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected a mapping payload for key '{record.key}', got {type(payload)!r}."
            )

        latents = payload.get("latents")
        label = payload.get("label")

        if not isinstance(latents, Tensor):
            raise TypeError(
                f"Expected 'latents' tensor for key '{record.key}', got {type(latents)!r}."
            )
        if not isinstance(label, Tensor):
            raise TypeError(
                f"Expected 'label' tensor for key '{record.key}', got {type(label)!r}."
            )

        return latents, label
