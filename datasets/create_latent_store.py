"""Utilities for exporting dataset latents with EnCodec."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torchaudio
from encodec import EncodecModel
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.poly_dataset import PolyphonicAsyncDataset
from datasets.store_utils import (
    DEFAULT_SAMPLES_PER_SHARD,
    SampleStoreWriter,
    write_global_metadata,
    write_index,
    write_shard_list,
)
from datasets.wav_midi_salience_dataset import WavMidiSalienceDataset

@dataclass
class _EncodedBatch:
    latents: Tensor
    labels: Tensor
    scales: Optional[Tensor]


def create_latent_store(
        dataset: PolyphonicAsyncDataset | WavMidiSalienceDataset,
        dataset_path: Union[str, Path],
        *,
        model_type: str = "24khz",
        target_bandwidth: float = 24.0,  # kbit/s
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD,
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

    writer = SampleStoreWriter(
        path,
        samples_per_shard,
        shard_pad,
        key_pad,
        shard_count,
        sample_key="latents",
    )

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

            print(f"Sample index {next_dataset_index}/{total_samples}")

            encoded_batch = _encode_batch(
                batch_items,
                model,
                dataset_sample_rate,
                eff_normalize,
            )

            extras = {"scale": encoded_batch.scales} if encoded_batch.scales is not None else None
            writer.write_batch(
                start_index=next_dataset_index,
                samples=encoded_batch.latents,
                labels=encoded_batch.labels,
                sample_callback=latent_callback,
                extras=extras,
            )

            next_dataset_index += batch_size

    if next_dataset_index != total_samples:
        raise RuntimeError("Encoded sample count mismatch.")

    shard_paths, global_index = writer.finalize()

    write_global_metadata(path, combined_metadata)
    write_index(path, global_index)
    write_shard_list(path, shard_paths)


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
