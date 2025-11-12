"""Utilities for converting MIDI data into salience tensors."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import mido
import torch


@dataclass
class _MidiEvent:
    time: float
    message: "mido.Message"
    sort_index: int


def _load_midi_events(midi_path: str) -> Tuple["mido.MidiFile", Sequence[_MidiEvent]]:
    midi = mido.MidiFile(midi_path)
    events: List[_MidiEvent] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_ticks = 0
        for msg in track:
            absolute_ticks += msg.time
            events.append(_MidiEvent(absolute_ticks, msg, track_index))
    events.sort(key=lambda evt: (evt.time, evt.sort_index))

    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000
    last_tick = 0
    current_time = 0.0
    real_time_events: List[_MidiEvent] = []
    for event in events:
        delta_ticks = event.time - last_tick
        if delta_ticks:
            current_time += mido.tick2second(delta_ticks, ticks_per_beat, tempo)
        last_tick = event.time
        if event.message.type == "set_tempo":
            tempo = event.message.tempo
            continue
        real_time_events.append(_MidiEvent(current_time, event.message, event.sort_index))

    return midi, tuple(real_time_events)


def _update_salience_segment(
        salience: torch.Tensor,
        start_sample: int,
        end_sample: int,
        active_notes: Sequence[int],
) -> None:
    if end_sample <= start_sample:
        return
    if not active_notes:
        return
    salience[start_sample:end_sample, active_notes] = 1.0


def _compute_activation(
        midi: "mido.MidiFile",
        events: Sequence[_MidiEvent],
        sample_rate: int,
        sustain_extend: bool,
) -> torch.Tensor:
    total_duration = max(midi.length, events[-1].time if events else 0.0)
    total_samples = max(int(round(total_duration * sample_rate)), 1)
    salience = torch.zeros((total_samples, 128), dtype=torch.float32)

    pressed_notes: set[int] = set()
    sustained_notes: set[int] = set()
    sustain_active = False
    last_time = 0.0

    for event in events:
        current_time = event.time
        start_sample = min(int(round(last_time * sample_rate)), total_samples)
        end_sample = min(int(round(current_time * sample_rate)), total_samples)
        active_now = sorted(pressed_notes | sustained_notes)
        _update_salience_segment(salience, start_sample, end_sample, active_now)

        msg = event.message
        if msg.type == "note_on" and msg.velocity == 0:
            msg = mido.Message("note_off", note=msg.note, time=msg.time)

        if msg.type == "note_on":
            note = int(msg.note)
            pressed_notes.add(note)
            if sustain_extend and note in sustained_notes:
                sustained_notes.discard(note)
        elif msg.type == "note_off":
            note = int(msg.note)
            pressed_notes.discard(note)
            if sustain_extend and sustain_active:
                sustained_notes.add(note)
            else:
                sustained_notes.discard(note)
        elif sustain_extend and msg.type == "control_change" and msg.control == 64:
            sustain_active = msg.value >= 64
            if not sustain_active:
                sustained_notes.clear()
        last_time = current_time

    active_now = sorted(pressed_notes | sustained_notes)
    start_sample = min(int(round(last_time * sample_rate)), total_samples)
    end_sample = total_samples
    _update_salience_segment(salience, start_sample, end_sample, active_now)

    return salience


def midi_to_salience(
        midi_path: str,
        *,
        chunk_duration: float = 30.0,
        frame_rate: int = 75,
        label_type: str = "activation",
        sustain_extend: bool = True,
        audio_path: str | None = None,
) -> List[torch.Tensor]:
    """Convert a MIDI file into salience tensors.

    Args:
        midi_path: Path to the MIDI file to load.
        chunk_duration: Duration of each chunk in seconds.
        frame_rate: Target sample rate for temporal resolution.
        label_type: Either ``"activation"`` or ``"power"``.
        sustain_extend: If ``True``, emulate sustain pedal behaviour.
        audio_path: Path to an audio file when ``label_type`` is ``"power"``.

    Returns:
        A list of tensors of shape ``(chunk_duration * sample_rate, 128)``.
    """

    midi, events = _load_midi_events(midi_path)
    if not events:
        return []

    activation = _compute_activation(midi, events, frame_rate, sustain_extend)

    frames_per_chunk = int(round(chunk_duration * frame_rate))
    total_frames = activation.shape[0] - (activation.shape[0] % frames_per_chunk)
    total_chunks = total_frames // frames_per_chunk
    if total_frames <= 0:
        return []
    activation = activation[:total_frames]

    if label_type == "activation":
        chunks = activation.view(total_chunks, frames_per_chunk, 128)
        return [chunk.clone() for chunk in chunks]

    if label_type != "power":
        raise ValueError("label_type must be either 'activation' or 'power'")

    if audio_path is None:
        raise ValueError("audio_path must be provided when label_type is 'power'")

    import numpy as np
    import librosa

    power_chunks: List[torch.Tensor] = []

    for i in range(total_chunks):
        offset = i * chunk_duration
        # pad window on both sides with half a frame
        audio_chunk, sr = librosa.load(
            audio_path,
            sr=None,
            mono=True,
            offset=offset,
            duration=chunk_duration,
        )

        hop_length = sr // frame_rate
        cqt = librosa.cqt(
            audio_chunk,
            sr=sr,
            hop_length=hop_length,
            fmin=librosa.midi_to_hz(0),
            n_bins=128,
            bins_per_octave=12,
        ).T
        # remove at most one time frame
        if cqt.shape[0] == frames_per_chunk + 1:
            cqt = cqt[:frames_per_chunk]
        else:
            raise ValueError(f"Expected cqt to have {frames_per_chunk + 1} frames, but was {cqt.shape[0]}")
        power = np.abs(cqt) ** 2
        chunk_tensor = torch.from_numpy(power)
        power_chunks.append(chunk_tensor.contiguous())

    if not power_chunks:
        return []

    power_tensor = torch.cat(power_chunks, dim=0)
    if power_tensor.shape[0] != total_frames:
        raise ValueError(f"Power tensor must have {total_frames} frames, but was {power_tensor.shape[0]}")

    masked = power_tensor * activation
    chunks = masked.view(total_chunks, frames_per_chunk, 128)
    return [chunk.clone() for chunk in chunks]
