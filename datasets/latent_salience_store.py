import io
import json
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import zstandard
from torch import Tensor
from torch.utils.data import Dataset


class LatentSalienceStore(Dataset[Tuple[Tensor, Tensor]]):
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
        latents, _, label = self._load_payload(record)

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

    def _load_index(self, index_path: Path) -> List["LatentSalienceStore._IndexRecord"]:
        records: List[LatentSalienceStore._IndexRecord] = []
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

    def _load_payload(self, record: "LatentSalienceStore._IndexRecord") -> Tuple[Tensor, float, Tensor]:
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
        scale = payload.get("scale")
        label = payload.get("label")

        if not isinstance(latents, Tensor):
            raise TypeError(
                f"Expected 'latents' tensor for key '{record.key}', got {type(latents)!r}."
            )
        if not isinstance(label, Tensor):
            raise TypeError(
                f"Expected 'label' tensor for key '{record.key}', got {type(label)!r}."
            )

        return latents, scale, label
