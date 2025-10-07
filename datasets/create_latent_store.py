"""Utilities for exporting dataset latents with EnCodec."""

from __future__ import annotations

import io
import json
import math
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import zstandard
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

DatasetItem = Tuple[Tensor, ...]

_SAMPLES_PER_SHARD = 10_000


@dataclass
class _MemberInfo:
    name: str
    tar_offset: int


def create_latent_store(
        dataset: TorchDataset[DatasetItem],
        dataset_path: Union[str, Path],
        dataset_sample_rate: int,
        target_bandwidth: float = 24.0,  # kbit/s
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
) -> None:
    """Encode a dataset to EnCodec pre-quant latents and persist them as a WebDataset.

    Args:
        dataset: ``torch.utils.data.Dataset`` providing items whose first element
            is a waveform tensor shaped ``(channels, samples)``. Additional
            elements are stored alongside the latent representation.
        dataset_path: Destination directory for the WebDataset shards. Parent
            directories are created automatically.
        dataset_sample_rate: Original sample rate of the dataset audio.
        device: Optional torch device for running the EnCodec encoder. Defaults
            to ``"cuda"`` when available otherwise ``"cpu"``.
        encoder: Optional pre-configured ``EncodecModel`` instance. When not
            provided the 24 kHz pretrained encoder shipped with the library is
            used.
        target_bandwidth: Optional bandwidth value passed to the encoder via
            ``set_target_bandwidth``. Unit is kbit/s, [1.5, 3, 6, 12, 24]
        metadata: Optional mapping of metadata describing the dataset creation
            context. These values are stored under the ``"external"`` key in the
            resulting dataset metadata file alongside encoding metadata recorded
            by this function.
    """

    from encodec import EncodecModel
    from encodec.utils import convert_audio

    if not isinstance(dataset, TorchDataset):
        raise TypeError("dataset must be an instance of torch.utils.data.Dataset.")

    try:
        total_samples = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive path
        raise TypeError("dataset must implement __len__ for deterministic shuffling.") from exc

    if total_samples <= 0:
        raise ValueError("dataset must contain at least one sample")

    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(target_bandwidth)
    model = model.to(device)
    model.eval()

    shard_count = math.ceil(total_samples / _SAMPLES_PER_SHARD)
    shard_pad = max(3, len(str(shard_count - 1)))
    key_pad = max(4, len(str(total_samples - 1)))

    encoding_metadata: Dict[str, Any] = {
        "length": total_samples,
        "dataset_sample_rate": int(dataset_sample_rate),
        "encoder_sample_rate": int(model.sample_rate),
        "encoder_channels": int(model.channels),
        "target_bandwidth": model.bandwidth,
        "dataset_type": type(dataset).__qualname__,
        "shard_size": _SAMPLES_PER_SHARD,
        "num_shards": shard_count,
        "key_width": key_pad,
        "shard_name_width": shard_pad,
        "created_unix": int(time.time()),
        "version": "1.0",
    }

    combined_metadata: Dict[str, Any] = {"encoding": encoding_metadata}
    if metadata is not None:
        combined_metadata["external"] = dict(metadata)

    shard_paths: List[str] = []
    global_index: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for shard_idx in range(shard_count):
            print(f"Shard {shard_idx + 1}/{shard_count}")
            start = shard_idx * _SAMPLES_PER_SHARD
            end = min(start + _SAMPLES_PER_SHARD, total_samples)

            shard_name = f"dataset-{shard_idx:0{shard_pad}d}"
            tar_path = path / f"{shard_name}.tar"
            zst_path = path / f"{shard_name}.tar.zst"
            manifest_path = path / f"{shard_name}.jsonl"

            manifest_records: List[Dict[str, Any]] = [
                {"__meta__": {"version": "1.0", "schema": "key,path,frequency"}}
            ]
            shard_samples: List[Dict[str, Any]] = []

            with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for dataset_index in range(start, end):
                    if dataset_index % 100 == 0:
                        print(f"Sample index {dataset_index}/{total_samples}")
                    item = dataset[dataset_index]  # type: ignore[index]
                    if not isinstance(item, tuple) or not item:
                        raise TypeError("Dataset items must be tuples with at least a waveform tensor.")

                    waveform = item[0]
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    if waveform.dim() != 2:
                        raise ValueError("Waveform tensor must have shape (channels, samples).")

                    waveform = waveform.to(torch.float32)
                    resampled = convert_audio(waveform, int(dataset_sample_rate), model.sample_rate, model.channels)
                    resampled = resampled.unsqueeze(0).to(device)

                    latents = model.encoder(resampled).squeeze(0).contiguous().to("cpu")

                    if len(item) != 2:
                        raise ValueError(
                            f"Dataset items must provide exactly one label (frequency) as the second element, but has length {len(item)}.")

                    frequency_tensor = item[1]
                    frequency_value = _to_float(frequency_tensor)

                    payload: Dict[str, Any] = {
                        "latent": latents,
                        "frequency": torch.tensor(frequency_value, dtype=torch.float32),
                    }

                    key = f"{dataset_index:0{key_pad}d}"
                    filename = f"{key}.pt"

                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    data = buffer.getvalue()

                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(data)
                    tarinfo.mtime = int(time.time())
                    tar.addfile(tarinfo, io.BytesIO(data))

                    manifest_records.append({
                        "key": key,
                        "path": filename,
                        "frequency": frequency_value,
                        "shape": list(latents.shape),
                        "dtype": str(latents.dtype),
                    })
                    shard_samples.append({
                        "key": key,
                        "path": filename,
                        "frequency": frequency_value,
                    })

            members = _read_tar_members(tar_path)
            compressed_members = _compress_tar_with_offsets(tar_path, zst_path, members)
            os.remove(tar_path)

            with manifest_path.open("w", encoding="utf-8") as manifest_file:
                for record in manifest_records:
                    manifest_file.write(json.dumps(record, separators=(",", ":")) + "\n")

            shard_paths.append(zst_path.name)
            if len(shard_samples) != len(compressed_members):
                raise RuntimeError("Shard sample count mismatch while building index.")
            for sample, member in zip(shard_samples, compressed_members):
                global_index.append({
                    "key": sample["key"],
                    "shard": zst_path.name,
                    "member": sample["path"],
                    "offset": member["offset"],
                    "size": member["size"],
                })

    _write_global_metadata(path, combined_metadata)
    _write_index(path, global_index)
    _write_shard_list(path, shard_paths)
    _write_shuffle_file(path, len(shard_paths))


def _write_global_metadata(destination: Path, metadata: Dict[str, Any]) -> None:
    metadata_path = destination / "dataset.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def _write_index(destination: Path, records: Iterable[Dict[str, Any]]) -> None:
    index_path = destination / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"__meta__": {"version": "1.0", "schema": "key,shard,member,offset,size"}},
                              separators=(",", ":")) + "\n")
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_shard_list(destination: Path, shard_paths: Sequence[str]) -> None:
    shard_list_path = destination / "shards.txt"
    with shard_list_path.open("w", encoding="utf-8") as file:
        for shard in shard_paths:
            file.write(f"{shard}\n")


def _write_shuffle_file(destination: Path, num_shards: int) -> None:
    shuffle_path = destination / "shuffle.npy"
    rng = np.random.default_rng(seed=0)
    permutation = rng.permutation(num_shards)
    np.save(shuffle_path, permutation, allow_pickle=False)


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


def _to_float(value: Any) -> float:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() != 1:
            raise ValueError("Frequency tensor must contain exactly one element.")
        return float(tensor.item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError("Frequency value must be a scalar tensor or numeric type.")


def _prepare_extras(extras: Sequence[Any]) -> Tuple[Any, ...]:
    converted: List[Any] = []
    for value in extras:
        if isinstance(value, Tensor):
            converted.append(value.detach().cpu())
        else:
            converted.append(value)
    return tuple(converted)


def _main() -> None:
    raise SystemExit("This module provides the 'create_latent_store' function. Import and call it directly.")


if __name__ == "__main__":
    _main()
