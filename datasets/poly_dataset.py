"""PyTorch dataset for generating asynchronous polyphonic audio clips."""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import zstandard

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


class PolyphonicAsyncDatasetFromStore(Dataset[Tuple[Tensor, Label]]):
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

        self._labels = self._load_labels(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Tuple[Tensor, Label]:
        if index < 0 or index >= len(self._records):
            raise IndexError("index out of range")

        record = self._records[index]
        latents = self._load_latents(record)
        labels = self._labels.get(record.key)
        if labels is None:
            raise KeyError(f"Missing labels for key '{record.key}'.")

        return latents, labels

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

    def _load_labels(
        self,
        records: Sequence["PolyphonicAsyncDatasetFromStore._IndexRecord"],
    ) -> Dict[str, Label]:
        labels: Dict[str, Label] = {}
        shards = {record.shard for record in records}

        for shard in shards:
            manifest_path = self._root / shard.replace(".tar.zst", ".jsonl")
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing manifest for shard '{shard}'.")

            with manifest_path.open("r", encoding="utf-8") as manifest:
                for line_number, line in enumerate(manifest):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "__meta__" in record:
                        continue

                    key = str(record.get("key"))
                    if not key:
                        raise ValueError(
                            f"Manifest '{manifest_path}' contains an entry without a key "
                            f"(line {line_number + 1})."
                        )

                    events = record.get("events", [])
                    labels[key] = [
                        (
                            float(event["frequency"]),
                            float(event["start"]),
                            float(event["end"]),
                        )
                        for event in events
                    ]

        return labels

    def _load_latents(self, record: "PolyphonicAsyncDatasetFromStore._IndexRecord") -> Tensor:
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
        tensor = torch.load(io.BytesIO(file_data), map_location=self._map_location)
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"Expected a torch.Tensor for key '{record.key}', got {type(tensor)!r}."
            )

        return tensor
