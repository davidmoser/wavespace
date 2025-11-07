"""DAC roundtrip script.

This script loads an audio file, encodes it with the Descript Audio Codec (DAC),
then decodes it and writes the reconstructed waveform next to the original
file. The output filename contains the chosen model configuration.
"""


from pathlib import Path
from typing import Optional

import torch
import torchaudio
from audiotools import AudioSignal
from dac.utils import load_model

MODEL_TYPES = ("16khz", "24khz", "44khz")
MODEL_BITRATES = ("8kbps", "16kbps")


def _load_model(
    model_type: str,
    model_bitrate: str,
    device: str,
    *,
    model_tag: str = "latest",
):
    model = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
    )
    model.to(device)
    model.eval()
    return model


def dac_roundtrip(
    input_path: Path,
    *,
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    model_tag: str = "latest",
    output_format: str = "wav",
    device: Optional[str] = None,
    win_duration: float = 60.0,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}; received {model_type!r}")

    if model_bitrate not in MODEL_BITRATES:
        raise ValueError(f"model_bitrate must be one of {MODEL_BITRATES}; received {model_bitrate!r}")

    if model_bitrate == "16kbps" and model_type != "44khz":
        raise ValueError("The 16kbps model is only available for the 44khz variant.")

    if output_format not in {"wav", "mp3"}:
        raise ValueError("output_format must be either 'wav' or 'mp3'")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    signal = AudioSignal(str(input_path))

    model = _load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        model_tag=model_tag,
        device=device,
    )

    # match the model SR
    if signal.sample_rate != model.sample_rate:
        print("Converting framerate")
        signal = signal.resample(model.sample_rate)

    # Workaround for DAC issue #80: ensure win_duration < signal duration
    sig_dur = signal.signal_duration
    safe_win = win_duration
    if safe_win >= sig_dur:
        safe_win = max(sig_dur - (1.0 / signal.sample_rate), 1.0 / signal.sample_rate)

    signal = signal.to(model.device)

    print("Compressing")

    with torch.inference_mode():
        dac_file = model.compress(
            signal,
            win_duration=safe_win,
        )
        print("Decompressing")
        reconstructed = model.decompress(dac_file).cpu()

    reconstructed = reconstructed.cpu()

    waveform = reconstructed.audio_data.squeeze(0)
    sample_rate = reconstructed.sample_rate

    suffix = f"_dac_{model_type}_{model_bitrate}"
    output_path = input_path.with_name(
        f"{input_path.stem}{suffix}.{output_format}"
    )

    print("Saving")
    if output_format == "wav":
        torchaudio.save(str(output_path), waveform, sample_rate)
    else:
        torchaudio.save(str(output_path), waveform, sample_rate, format="mp3")

    return output_path


if __name__ == "__main__":
    input_file = Path("../resources/sine/sine_1000Hz_1s_16kHz.wav")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    result_path = dac_roundtrip(
        input_file,
        model_type="44khz", # 16, 24, 44
        model_bitrate="8kbps", # 8, 16
        output_format="wav",
        device=None,
    )
    print(f"Saved decoded audio to {result_path}")
