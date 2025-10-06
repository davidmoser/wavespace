"""Utilities for exporting dataset latents with EnCodec."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterator, MutableMapping, Optional, Tuple, Union

import lmdb
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset, IterableDataset

DatasetItem = Tuple[Tensor, ...]


def create_latent_store(
        dataset: TorchDataset[DatasetItem],
        lmdb_path: Union[str, Path],
        *,
        sample_rate: Optional[int] = None,
        map_size_bytes: int = 1 << 33,
        device: Optional[torch.device] = None,
        encoder: Optional["EncodecModel"] = None,
        target_bandwidth: Optional[float] = None,
) -> None:
    """Encode a dataset to EnCodec pre-quant latents and persist them in LMDB.

    Args:
        dataset: ``torch.utils.data.Dataset`` providing items whose first element
            is a waveform tensor shaped ``(channels, samples)``. Additional
            elements are stored alongside the latent representation.
        lmdb_path: Destination path of the LMDB database. Parent directories are
            created automatically.
        sample_rate: Original sample rate of the dataset audio. When omitted the
            function attempts to read a ``sample_rate`` attribute from the
            dataset instance.
        map_size_bytes: Initial LMDB map size in bytes. The map size grows
            automatically if it becomes insufficient while writing samples.
        device: Optional torch device for running the EnCodec encoder. Defaults
            to ``"cuda"`` when available otherwise ``"cpu"``.
        encoder: Optional pre-configured ``EncodecModel`` instance. When not
            provided the 24 kHz pretrained encoder shipped with the library is
            used.
        target_bandwidth: Optional bandwidth value passed to the encoder via
            ``set_target_bandwidth``.
    """

    from encodec import EncodecModel
    from encodec.utils import convert_audio

    path = Path(lmdb_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = encoder or EncodecModel.encodec_model_24khz()
    if target_bandwidth is not None:
        model.set_target_bandwidth(target_bandwidth)
    model = model.to(device)
    model.eval()

    if not isinstance(dataset, TorchDataset):
        raise TypeError("dataset must be an instance of torch.utils.data.Dataset.")

    dataset_sample_rate = sample_rate if sample_rate is not None else getattr(dataset, "sample_rate", None)
    if dataset_sample_rate is None:
        raise ValueError("Dataset sample rate must be provided explicitly or via a 'sample_rate' attribute.")

    length = None
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        pass

    env = lmdb.open(str(path), map_size=map_size_bytes)
    try:
        metadata: MutableMapping[str, Any] = {
            "length": length,
            "dataset_sample_rate": int(dataset_sample_rate),
            "encoder_sample_rate": int(model.sample_rate),
            "encoder_channels": int(model.channels),
            "target_bandwidth": model.bandwidth,
            "dataset_type": type(dataset).__qualname__,
        }
        _put_with_resize(env, b"__metadata__", pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))

        with torch.inference_mode():
            for index, item in enumerate(_iterate_dataset(dataset)):
                if not item:
                    raise ValueError("Dataset items must contain at least a waveform tensor.")

                waveform = item[0]
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                if waveform.dim() != 2:
                    raise ValueError("Waveform tensor must have shape (channels, samples).")

                waveform = waveform.to(torch.float32)
                resampled = convert_audio(waveform, int(dataset_sample_rate), model.sample_rate, model.channels)
                resampled = resampled.unsqueeze(0).to(device)

                latents = model.encoder(resampled).squeeze(0).cpu()

                payload: dict[str, Any] = {"latent": latents}
                if len(item) > 1:
                    extras = tuple(obj for obj in item[1:])
                    payload["extras"] = extras if len(extras) > 1 else extras[0]

                key = f"{index:08d}".encode("utf-8")
                value_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                _put_with_resize(env, key, value_bytes)
    finally:
        env.close()


def _iterate_dataset(dataset: TorchDataset[DatasetItem]) -> Iterator[DatasetItem]:
    if isinstance(dataset, IterableDataset):
        for item in dataset:
            if not isinstance(item, tuple):
                raise TypeError("IterableDataset items must be tuples of tensors.")
            yield item
        return

    length = len(dataset)  # type: ignore[arg-type]
    for index in range(length):
        item = dataset[index]  # type: ignore[index]
        if not isinstance(item, tuple):
            raise TypeError("Dataset items must be tuples when accessed by index.")
        yield item


def _put_with_resize(env: lmdb.Environment, key: bytes, value: bytes) -> None:
    while True:
        try:
            with env.begin(write=True) as txn:
                txn.put(key, value)
            break
        except lmdb.MapFullError:
            current = env.info()["map_size"]
            env.set_mapsize(max(current * 2, current + len(value)))


def _main() -> None:
    raise SystemExit("This module provides the 'create_latent_store' function. Import and call it directly.")


if __name__ == "__main__":
    _main()
