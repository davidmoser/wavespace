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
    sample_rate: int = 16000,
    label_type: str = "activation",
    sustain_extend: bool = True,
    audio_path: str | None = None,
) -> List[torch.Tensor]:
    """Convert a MIDI file into salience tensors.

    Args:
        midi_path: Path to the MIDI file to load.
        chunk_duration: Duration of each chunk in seconds.
        sample_rate: Target sample rate for temporal resolution.
        label_type: Either ``"activation"`` or ``"power"``.
        sustain_extend: If ``True``, emulate sustain pedal behaviour.
        audio_path: Path to an audio file when ``label_type`` is ``"power"``.

    Returns:
        A list of tensors of shape ``(chunk_duration * sample_rate, 128)``.
    """
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    midi, events = _load_midi_events(midi_path)
    if not events:
        return []

    activation = _compute_activation(midi, events, sample_rate, sustain_extend)

    chunk_frames = int(round(chunk_duration * sample_rate))
    if chunk_frames <= 0:
        raise ValueError("chunk_duration and sample_rate combination is invalid")

    if label_type == "activation":
        total_frames = activation.shape[0] - (activation.shape[0] % chunk_frames)
        if total_frames <= 0:
            return []
        activation = activation[:total_frames]
        chunks = activation.view(total_frames // chunk_frames, chunk_frames, 128)
        return [chunk.clone() for chunk in chunks]

    if label_type != "power":
        raise ValueError("label_type must be either 'activation' or 'power'")

    if audio_path is None:
        raise ValueError("audio_path must be provided when label_type is 'power'")

    import numpy as np
    import librosa

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.ndim != 1:
        audio = librosa.to_mono(audio)

    cqt = librosa.cqt(
        audio,
        sr=sample_rate,
        hop_length=1,
        fmin=librosa.midi_to_hz(0),
        n_bins=128,
        bins_per_octave=12,
    )
    power = np.abs(cqt) ** 2
    power_tensor = torch.from_numpy(power.T.astype(np.float32))

    max_frames = min(power_tensor.shape[0], activation.shape[0])
    if max_frames < chunk_frames:
        return []
    activation = activation[:max_frames]
    power_tensor = power_tensor[:max_frames]
    masked = power_tensor * activation

    total_frames = max_frames - (max_frames % chunk_frames)
    if total_frames <= 0:
        return []
    masked = masked[:total_frames]
    chunks = masked.view(total_frames // chunk_frames, chunk_frames, 128)
    return [chunk.clone() for chunk in chunks]
