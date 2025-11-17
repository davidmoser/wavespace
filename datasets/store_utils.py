"""Shared helpers for writing dataset shards to disk."""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import zstandard
from torch import Tensor

DEFAULT_SAMPLES_PER_SHARD = 10_000


@dataclass
class _MemberInfo:
    name: str
    tar_offset: int


class SampleStoreWriter:
    """Writer that serializes sample/label tensors into WebDataset shards."""

    def __init__(
            self,
            destination: Path,
            samples_per_shard: int,
            shard_pad: int,
            key_pad: int,
            total_shards: int,
            *,
            sample_key: str,
    ) -> None:
        self._destination = destination
        self._samples_per_shard = samples_per_shard
        self._shard_pad = shard_pad
        self._key_pad = key_pad
        self._total_shards = total_shards
        self._sample_key = sample_key

        self._current_shard_idx: Optional[int] = None
        self._current_tar: Optional[tarfile.TarFile] = None
        self._current_tar_path: Optional[Path] = None
        self._current_zst_path: Optional[Path] = None
        self._current_shard_samples: List[Dict[str, Any]] = []

        self._shard_paths: List[str] = []
        self._global_index: List[Dict[str, Any]] = []

    def write_batch(
            self,
            *,
            start_index: int,
            samples: Tensor,
            labels: Tensor,
            sample_callback: Optional[Callable[[int, Tensor], None]] = None,
            extras: Optional[Dict[str, Tensor]] = None,
    ) -> None:

        if not isinstance(samples, Tensor):
            raise TypeError("Samples must be provided as a torch.Tensor batch.")
        if not isinstance(labels, Tensor):
            raise TypeError("Labels must be provided as a torch.Tensor batch.")

        batch_size = int(samples.shape[0]) if samples.dim() > 0 else 1
        if labels.shape[0] != batch_size:
            raise ValueError("Sample and label batch sizes must match.")

        extras = extras or {}
        for name, tensor in extras.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Extra '{name}' must be a torch.Tensor.")
            if tensor.shape[0] != batch_size:
                raise ValueError(f"Extra '{name}' must match batch size {batch_size}.")

        for offset in range(batch_size):
            dataset_index = start_index + offset
            self._ensure_shard(dataset_index)

            sample_tensor = samples[offset].detach().to(torch.float32).contiguous().cpu()
            if sample_callback is not None:
                sample_callback(dataset_index, sample_tensor)

            payload: Dict[str, Any] = {self._sample_key: sample_tensor}

            label_tensor = labels[offset].detach().to(torch.float32).contiguous().cpu()
            payload["label"] = label_tensor

            for name, tensor in extras.items():
                payload[name] = tensor[offset].detach().to(torch.float32).contiguous().cpu()

            key = f"{dataset_index:0{self._key_pad}d}"
            filename = f"{key}.pt"

            buffer = io.BytesIO()
            torch.save(payload, buffer)
            data = buffer.getvalue()

            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tarinfo.mtime = int(time.time())
            if self._current_tar is None:
                raise RuntimeError("Tarfile handle is not initialized.")
            self._current_tar.addfile(tarinfo, io.BytesIO(data))

            self._current_shard_samples.append({"key": key, "path": filename})

    def finalize(self) -> tuple[list[str], list[dict[str, Any]]]:
        self._close_current_shard()
        return self._shard_paths, self._global_index

    def _ensure_shard(self, dataset_index: int) -> None:
        shard_idx = dataset_index // self._samples_per_shard
        if self._current_shard_idx == shard_idx:
            return

        self._close_current_shard()
        self._open_shard(shard_idx)

    def _open_shard(self, shard_idx: int) -> None:
        self._current_shard_idx = shard_idx
        shard_name = f"dataset-{shard_idx:0{self._shard_pad}d}"
        print(f"Shard {shard_idx + 1}/{self._total_shards}")
        self._current_tar_path = self._destination / f"{shard_name}.tar"
        self._current_zst_path = self._destination / f"{shard_name}.tar.zst"
        self._current_tar = tarfile.open(self._current_tar_path, mode="w", format=tarfile.PAX_FORMAT)
        self._current_shard_samples = []

    def _close_current_shard(self) -> None:
        if self._current_tar is None:
            return

        self._current_tar.close()
        if self._current_tar_path is None or self._current_zst_path is None:
            raise RuntimeError("Shard paths are not initialized.")

        members = _read_tar_members(self._current_tar_path)
        compressed_members = _compress_tar_with_offsets(
            self._current_tar_path,
            self._current_zst_path,
            members,
        )
        os.remove(self._current_tar_path)

        self._shard_paths.append(self._current_zst_path.name)
        if len(self._current_shard_samples) != len(compressed_members):
            raise RuntimeError("Shard sample count mismatch while building index.")

        for sample, member in zip(self._current_shard_samples, compressed_members):
            self._global_index.append({
                "key": sample["key"],
                "shard": self._current_zst_path.name,
                "member": sample["path"],
                "offset": member["offset"],
                "size": member["size"],
            })

        self._current_tar = None
        self._current_tar_path = None
        self._current_zst_path = None
        self._current_shard_samples = []
        self._current_shard_idx = None


def write_global_metadata(destination: Path, metadata: Dict[str, Any]) -> None:
    metadata_path = destination / "dataset.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def write_index(destination: Path, records: Iterable[Dict[str, Any]]) -> None:
    index_path = destination / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"__meta__": {"version": "1.0", "schema": "key,shard,member,offset,size"}},
                              separators=(",", ":")) + "\n")
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_shard_list(destination: Path, shard_paths: Sequence[str]) -> None:
    shard_list_path = destination / "shards.txt"
    with shard_list_path.open("w", encoding="utf-8") as file:
        for shard in shard_paths:
            file.write(f"{shard}\n")


def _read_tar_members(tar_path: Path) -> List[_MemberInfo]:
    members: List[_MemberInfo] = []
    with tarfile.open(tar_path, mode="r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                members.append(_MemberInfo(name=member.name, tar_offset=member.offset))
    return members


def _compress_tar_with_offsets(tar_path: Path, zst_path: Path, members: Sequence[_MemberInfo]) -> List[Dict[str, int]]:
    if not members:
        return []

    tar_size = tar_path.stat().st_size
    ordered_members = list(sorted(members, key=lambda item: item.tar_offset))

    cctx = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    )

    offsets: Dict[str, int] = {}
    with tar_path.open("rb") as source, zst_path.open("wb") as destination:
        writer = cctx.stream_writer(destination)
        tar_position = 0
        index = 0

        chunk_size = 1 << 20
        while tar_position < tar_size:
            target_offset = ordered_members[index].tar_offset if index < len(ordered_members) else tar_size

            if tar_position == target_offset and index < len(ordered_members):
                writer.flush(zstandard.FLUSH_FRAME)
                offsets[ordered_members[index].name] = destination.tell()
                index += 1
                continue

            next_target = ordered_members[index].tar_offset if index < len(ordered_members) else tar_size
            bytes_remaining = next_target - tar_position
            to_read = min(chunk_size, bytes_remaining if bytes_remaining > 0 else tar_size - tar_position)
            chunk = source.read(to_read)
            if not chunk:
                break
            writer.write(chunk)
            tar_position += len(chunk)

        writer.flush(zstandard.FLUSH_FRAME)
        writer.close()

    compressed_size = zst_path.stat().st_size

    results: List[Dict[str, int]] = []
    ordered_names = [member.name for member in ordered_members]
    offsets_in_order = [offsets[name] for name in ordered_names]
    offsets_in_order.append(compressed_size)
    for idx, name in enumerate(ordered_names):
        start = offsets_in_order[idx]
        end = offsets_in_order[idx + 1]
        results.append({"member": name, "offset": start, "size": end - start})
    return results


__all__ = [
    "DEFAULT_SAMPLES_PER_SHARD",
    "SampleStoreWriter",
    "write_global_metadata",
    "write_index",
    "write_shard_list",
]
