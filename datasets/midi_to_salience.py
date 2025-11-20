"""Utilities for converting MIDI data into salience tensors."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mido
import torch
import torchaudio.functional as taf


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
        cqts: Optional[Sequence[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """Convert a MIDI file into salience tensors.

    Args:
        midi_path: Path to the MIDI file to load.
        chunk_duration: Duration of each chunk in seconds.
        frame_rate: Target sample rate for temporal resolution.
        label_type: Either ``"activation"`` or ``"power"``.
        sustain_extend: If ``True``, emulate sustain pedal behaviour.
        cqts: Pre-computed CQT power tensors when ``label_type`` is ``"power"``.

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

    if cqts is None:
        raise ValueError("cqts must be provided when label_type is 'power'")

    # sometimes MIDI ends a few seconds before wav => there might be one less salience chunk => drop wav chunk
    if len(cqts) == total_chunks + 1:
        cqts = cqts[:-1]

    if len(cqts) != total_chunks:
        raise ValueError(
            f"CQT chunk count ({len(cqts)}) must match MIDI chunk count ({total_chunks}).",
        )

    power_chunks = [chunk.to(dtype=torch.float32).contiguous() for chunk in cqts]
    power_tensor = torch.cat(power_chunks, dim=0)
    if power_tensor.shape[0] != total_frames:
        raise ValueError(
            f"Power tensor must have {total_frames} frames, but was {power_tensor.shape[0]}",
        )

    masked = power_tensor * activation
    chunks = masked.view(total_chunks, frames_per_chunk, 128)
    return [chunk.clone() for chunk in chunks]


def prepare_cqts(
        audio_path: str,
        *,
        chunk_duration: float,
        frame_rate: int,
) -> List[torch.Tensor]:
    """Compute CQT power tensors for an audio file."""

    import librosa
    import numpy as np
    import soundfile as sf

    # librosa is much faster with hop_length = integer * 2**k
    # with framerates like 10, 20 that's much better for 48 kHz as opposed to 44.1 kHz (10 times faster)
    optimal_sr = 48_000
    hop_length = optimal_sr // frame_rate

    with sf.SoundFile(audio_path) as audio_file:
        sr = int(audio_file.samplerate)
        total_samples = int(audio_file.frames)

    chunk_samples = int(round(chunk_duration * sr))
    frames_per_chunk = int(round(chunk_duration * frame_rate))
    if chunk_samples <= 0 or frames_per_chunk <= 0:
        return []

    usable_samples = total_samples - (total_samples % chunk_samples)
    if usable_samples < chunk_samples:
        return []

    total_chunks = usable_samples // chunk_samples

    cqt_chunks: List[torch.Tensor] = []
    with sf.SoundFile(audio_path) as audio_file:
        for chunk_index in range(total_chunks):
            audio_file.seek(chunk_index * chunk_samples)
            samples = audio_file.read(chunk_samples, dtype="float32", always_2d=True)
            if samples.shape[0] != chunk_samples:
                break
            chunk = samples.mean(axis=1)
            chunk_resampled = taf.resample(torch.from_numpy(chunk), orig_freq=sr, new_freq=optimal_sr).cpu().numpy()
            cqt = librosa.cqt(
                np.ascontiguousarray(chunk_resampled),
                sr=optimal_sr,
                hop_length=hop_length,
                fmin=librosa.midi_to_hz(0),
                n_bins=128 * 2,
                bins_per_octave=12 * 2,
            ).T
            # remove at most one time frame
            if cqt.shape[0] == frames_per_chunk + 1:
                cqt = cqt[:frames_per_chunk]
            elif cqt.shape[0] != frames_per_chunk:
                raise ValueError(
                    f"Expected cqt to have {frames_per_chunk}±1 frames, but was {cqt.shape[0]}",
                )
            power = (np.abs(cqt) ** 2).astype(np.float32, copy=False)
            cqt_chunks.append(torch.from_numpy(power).contiguous())

    if len(cqt_chunks) != total_chunks:
        raise ValueError(
            f"Expected {total_chunks} CQT chunks but generated {len(cqt_chunks)} for {audio_path}",
        )

    return cqt_chunks
