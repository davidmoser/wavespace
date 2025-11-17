"""Iterable dataset that pairs WAV audio chunks with MIDI-derived salience tensors."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from datasets.midi_to_salience import midi_to_salience, prepare_cqts

_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".wave")
_MIDI_EXTENSIONS: Tuple[str, ...] = (".mid", ".midi")


class WavMidiSalienceDataset(IterableDataset[Tuple[Tensor, Tensor]]):
    """Dataset that yields audio and MIDI salience tensor pairs."""

    def __init__(
            self,
            *,
            wav_midi_path: str | Path,
            n_samples: int,
            duration: float,
            sample_rate: int,
            label_frame_rate: float,
            label_type: str = "activation",
            seed: int = 20,
    ) -> None:
        self.root = Path(wav_midi_path).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"wav_midi_path must be a directory: {self.root}")

        self.n_samples = int(n_samples)
        self.duration = float(duration)
        self.sample_rate = int(sample_rate)
        self.label_frame_rate = float(label_frame_rate)

        label_type_value = str(label_type).lower()
        if label_type_value not in {"power", "activation"}:
            raise ValueError("label_type must be either 'power' or 'activation'")
        self.label_type = label_type_value

        self._base_files = self._collect_wav_files(self.root)
        random.seed(seed)
        random.shuffle(self._base_files)
        self._base_files = tuple(self._base_files)

        if not self._base_files:
            raise ValueError(f"No WAV files found under {self.root}")

        self._frame_rate = int(round(self.label_frame_rate))

    def __len__(self) -> int:  # pragma: no cover - optional for IterableDataset
        return self.n_samples

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        worker = get_worker_info()
        if worker is None:
            files = self._base_files
            target_samples = self.n_samples
        else:
            files = tuple(self._base_files[worker.id:: worker.num_workers])
            base_share = self.n_samples // worker.num_workers
            remainder = self.n_samples % worker.num_workers
            target_samples = base_share + (1 if worker.id < remainder else 0)

        if target_samples <= 0 or not files:
            return

        produced = 0
        pointer = 0

        audio_buffer: List[Tensor] = []
        label_buffer: List[Tensor] = []
        buffer_index = 0

        while produced < target_samples:
            if buffer_index >= len(audio_buffer):
                audio_buffer = []
                label_buffer = []
                buffer_index = 0

                attempts = 0
                max_attempts = len(files)
                while not audio_buffer and attempts < max_attempts:
                    if pointer >= len(files):
                        pointer = 0
                    wav_path = files[pointer]
                    pointer += 1
                    audio_buffer, label_buffer = self._prepare_file_chunks(wav_path)
                    attempts += 1

                if not audio_buffer:
                    break

            audio_chunk = audio_buffer[buffer_index]
            label_chunk = label_buffer[buffer_index]
            buffer_index += 1
            produced += 1
            yield audio_chunk, label_chunk

    def _collect_wav_files(self, root: Path) -> List[Path]:
        wav_files: List[Path] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for candidate in sorted(child.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in _AUDIO_EXTENSIONS:
                    wav_files.append(candidate)
        return wav_files

    def _prepare_file_chunks(self, wav_path: Path) -> Tuple[List[Tensor], List[Tensor]]:
        # Label chunks
        midi_path = self._resolve_midi_path(wav_path)
        cqts: List[Tensor] | None = None
        if self.label_type == "power":
            cqts = prepare_cqts(
                audio_path=str(wav_path),
                chunk_duration=self.duration,
                frame_rate=self._frame_rate,
            )
            if not cqts:
                return [], []

        salience_chunks = midi_to_salience(
            midi_path=str(midi_path),
            chunk_duration=self.duration,
            frame_rate=self._frame_rate,
            label_type=self.label_type,
            cqts=cqts,
        )
        if not salience_chunks:
            return [], []
        salience_chunks = [chunk.to(dtype=torch.float32).contiguous() for chunk in salience_chunks]

        # Audio chunks
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.mean(dim=0, keepdim=True)  # to mono
        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        total_samples = waveform.shape[1]
        chunk_samples = int(self.sample_rate * self.duration)
        usable_samples = total_samples - (total_samples % chunk_samples)
        if usable_samples < chunk_samples:
            return [], []

        waveform = waveform[:, :usable_samples]
        audio_chunks = [chunk.contiguous() for chunk in waveform.split(chunk_samples, dim=1)]

        # sometimes MIDI ends a few seconds before wav => there might be one less salience chunk => drop wav chunk
        if len(audio_chunks) == len(salience_chunks) + 1:
            audio_chunks = audio_chunks[:-1]
            print(f"Throwing away last audio chunk for:\n{wav_path}")

        if len(audio_chunks) != len(salience_chunks):
            raise Exception(
                f"Audio chunks ({len(audio_chunks)}) and salience chunks ({len(salience_chunks)}) don't match.\nFile: {wav_path}.")

        return audio_chunks, salience_chunks

    def _resolve_midi_path(self, wav_path: Path) -> Path:
        stem = wav_path.stem
        parent = wav_path.parent

        for ext in _MIDI_EXTENSIONS:
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            upper = candidate.with_suffix(ext.upper())
            if upper.exists():
                return upper

        for item in parent.iterdir():
            if item.is_file() and item.suffix.lower() in _MIDI_EXTENSIONS and item.stem == stem:
                return item

        raise FileNotFoundError(f"No MIDI file matching {wav_path.name} found in {parent}")

    def get_sample_rate(self) -> int:
        return self.sample_rate
