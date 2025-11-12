"""Utilities for exporting dataset latents with EnCodec."""

import io
import json
import math
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
import zstandard
from encodec import EncodecModel
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.poly_dataset import PolyphonicAsyncDataset
from datasets.wav_midi_salience_dataset import WavMidiSalienceDataset

_DEFAULT_SAMPLES_PER_SHARD = 10_000


@dataclass
class _MemberInfo:
    name: str
    tar_offset: int


def create_latent_store(
        dataset: PolyphonicAsyncDataset | WavMidiSalienceDataset,
        dataset_path: Union[str, Path],
        *,
        model_type: str = "24khz",
        target_bandwidth: float = 24.0,  # kbit/s
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        samples_per_shard: int = _DEFAULT_SAMPLES_PER_SHARD,
        encode_batch_size: int = 8,
        num_workers: int = 1,
        normalize: Optional[bool] = None,
        latent_callback: Optional[Callable[[int, Tensor], None]] = None,
) -> None:
    """Encode a dataset to EnCodec pre-quant latents and persist them as a WebDataset.

    Args:
        dataset: ``torch.utils.data.Dataset`` providing items whose first element
            is a waveform tensor shaped ``(channels, samples)``. Additional
            elements are stored alongside the latent representation.
        dataset_path: Destination directory for the WebDataset shards. Parent
            directories are created automatically.
        model_type: Type of the encodec model, "24khz" or "48khz".
        device: Optional torch device for running the EnCodec encoder. Defaults
            to ``"cuda"`` when available otherwise ``"cpu"``.
        target_bandwidth: Optional bandwidth value passed to the encoder via
            ``set_target_bandwidth``. Unit is kbit/s, [1.5, 3, 6, 12, 24]
        metadata: Optional mapping of metadata describing the dataset creation
            context. These values are stored under the ``"external"`` key in the
            resulting dataset metadata file alongside encoding metadata recorded
            by this function.
        samples_per_shard: Optional number of samples to include in each shard of
            the generated store. Defaults to 10,000 samples per shard.
        encode_batch_size: Optional number of samples processed per encoder call.
            Defaults to 8 samples per batch.
        latent_callback: Optional callable invoked with ``(dataset_index,
            latents, item)`` immediately after encoding each sample. This can be
            used by tests to inspect the generated latents without modifying the
            storage pipeline.
        normalize: Optional flag indicating whether to scale normalize the audio
            samples within segments (24kHz model is unnormalized by default,
            48kHz model is normalized within 1s segments by default)
        num_workers: Number of workers for the DataLoader
    """
    total_samples = len(dataset)  # type: ignore[arg-type]

    dataset_sample_rate = dataset.get_sample_rate()

    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be a positive integer.")
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be a positive integer.")

    model = None
    match model_type.lower():
        case "24khz":
            model = EncodecModel.encodec_model_24khz()
        case "48khz":
            model = EncodecModel.encodec_model_48khz()
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    model.set_target_bandwidth(target_bandwidth)
    model = model.to(device)
    model.eval()

    eff_normalize = normalize if normalize is not None else model.normalize

    shard_count = math.ceil(total_samples / samples_per_shard)
    shard_pad = max(3, len(str(shard_count - 1)))
    key_pad = max(4, len(str(total_samples - 1)))

    encoding_metadata: Dict[str, Any] = {
        "length": total_samples,
        "dataset_sample_rate": int(dataset_sample_rate),
        "encoder_sample_rate": int(model.sample_rate),
        "encoder_channels": int(model.channels),
        "model_type": model_type,
        "target_bandwidth": model.bandwidth,
        "dataset_type": type(dataset).__qualname__,
        "shard_size": int(samples_per_shard),
        "num_shards": shard_count,
        "key_width": key_pad,
        "shard_name_width": shard_pad,
        "created_unix": int(time.time()),
        "version": "1.0",
    }

    combined_metadata: Dict[str, Any] = {"encoding": encoding_metadata}
    if metadata is not None:
        combined_metadata["external"] = dict(metadata)

    workers = 0 if num_workers is None else num_workers
    loader = DataLoader(
        dataset,
        batch_size=encode_batch_size,
        num_workers=workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    writer = _LatentStoreWriter(path, samples_per_shard, shard_pad, key_pad, shard_count)

    next_dataset_index = 0

    with torch.inference_mode():
        for batch_items in loader:
            batch_waveforms = batch_items[0]
            if isinstance(batch_waveforms, Tensor):
                batch_size = int(batch_waveforms.shape[0]) if batch_waveforms.dim() > 0 else 1
            else:
                batch_size = len(batch_waveforms)

            if batch_size == 0:
                continue

            if next_dataset_index % 100 == 0:
                print(f"Sample index {next_dataset_index}/{total_samples}")

            encoded_batch = _encode_batch(
                batch_items,
                model,
                dataset_sample_rate,
                eff_normalize,
            )

            writer.write_batch(
                start_index=next_dataset_index,
                encoded_batch=encoded_batch,
                latent_callback=latent_callback,
            )

            next_dataset_index += batch_size

    if next_dataset_index != total_samples:
        raise RuntimeError("Encoded sample count mismatch.")

    shard_paths, global_index = writer.finalize()

    _write_global_metadata(path, combined_metadata)
    _write_index(path, global_index)
    _write_shard_list(path, shard_paths)


@dataclass
class _EncodedBatch:
    latents: Tensor
    labels: Tensor
    scales: Optional[Tensor]


def _encode_batch(
        batch_items: Tuple[Any, ...],
        model: EncodecModel,
        dataset_sample_rate: int,
        normalize: bool,
) -> _EncodedBatch:
    if not batch_items:
        raise ValueError("Batch is empty.")

    waveform_batch = batch_items[0]
    if not isinstance(waveform_batch, Tensor):
        raise TypeError("Waveform batch must be a torch.Tensor.")

    if waveform_batch.dim() != 3:
        raise ValueError("Waveform batch must have shape (batch, channels, samples)")
    if waveform_batch.shape[1] > 2:
        raise ValueError(f"Waveform batch must be mono or stereo. Channels {waveform_batch.shape[1]}")

    mono_waveform = waveform_batch.mean(dim=1, keepdim=True)
    resampler = torchaudio.transforms.Resample(int(dataset_sample_rate), int(model.sample_rate))
    resampled_waveforms = resampler(mono_waveform)

    scales: Optional[Tensor] = None
    if normalize:
        volume = resampled_waveforms.pow(2).mean(dim=1, keepdim=True).sqrt()
        scale = volume + 1e-8
        resampled_waveforms = resampled_waveforms / scale
        scales = scale.detach().cpu()

    latents = model.encoder(resampled_waveforms).contiguous().cpu()

    label_batch = batch_items[1]
    if not isinstance(label_batch, Tensor):
        raise TypeError("Label batch must be a torch.Tensor.")
    label_tensor = label_batch.detach().to(torch.float32).contiguous().cpu()

    encoded = _EncodedBatch(latents=latents, labels=label_tensor, scales=scales)
    return encoded


class _LatentStoreWriter:
    def __init__(
            self,
            destination: Path,
            samples_per_shard: int,
            shard_pad: int,
            key_pad: int,
            total_shards: int,
    ) -> None:
        self._destination = destination
        self._samples_per_shard = samples_per_shard
        self._shard_pad = shard_pad
        self._key_pad = key_pad
        self._total_shards = total_shards

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
            encoded_batch: _EncodedBatch,
            latent_callback: Optional[Callable[[int, Tensor], None]],
    ) -> None:
        batch_size = encoded_batch.latents.shape[0]

        for offset in range(batch_size):
            dataset_index = start_index + offset
            self._ensure_shard(dataset_index)

            sample_latents = encoded_batch.latents[offset].contiguous()

            if latent_callback is not None:
                latent_callback(dataset_index, sample_latents)

            payload: Dict[str, Any] = {"latents": sample_latents}

            if encoded_batch.scales is not None:
                scale = encoded_batch.scales[offset].view(-1, 1)
                payload["scale"] = scale

            label = encoded_batch.labels[offset]
            payload["label"] = label

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

    def finalize(self) -> Tuple[List[str], List[Dict[str, Any]]]:
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
