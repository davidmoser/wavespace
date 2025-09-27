"""Encodec roundtrip script.

This script loads an audio file, encodes it with the Encodec codec from Meta,
then decodes it and writes the reconstructed waveform next to the original
file. The output filename contains the chosen quality level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


QUALITY_LEVELS = [1.5, 3.0, 6.0, 12.0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and decode an audio file with the Encodec codec",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=6.0,
        choices=QUALITY_LEVELS,
        help=(
            "Target bandwidth in kbps used for Encodec's encode/decode cycle. "
            "This value controls the quality of the reconstruction and will "
            "also be embedded in the output filename."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (defaults to CUDA when available).",
    )
    parser.add_argument(
        "--model",
        choices=["24khz", "48khz"],
        default="24khz",
        help="Which pretrained Encodec model to use.",
    )
    parser.add_argument(
        "--output-format",
        choices=["wav", "mp3"],
        default="wav",
        help="Audio format for the decoded output file.",
    )
    return parser.parse_args()


def _load_model(model_name: str, device: str) -> EncodecModel:
    if model_name == "48khz":
        model = EncodecModel.encodec_model_48khz()
    else:
        model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidths(QUALITY_LEVELS)
    model.to(device)
    model.set_device(device)
    return model


def main() -> None:
    args = _parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    waveform, sample_rate = torchaudio.load(str(input_path))

    device = args.device
    model = _load_model(args.model, device)
    model.eval()

    # Prepare audio for the model
    waveform = waveform.to(device)
    audio = convert_audio(
        waveform,
        sample_rate,
        model.sample_rate,
        model.channels,
    ).unsqueeze(0)

    with torch.inference_mode():
        encoded = model.encode(audio, target_bandwidth=args.quality)
        decoded = model.decode(encoded)

    # Convert back to the original sample rate and channel count
    decoded = convert_audio(
        decoded.squeeze(0).cpu(),
        model.sample_rate,
        sample_rate,
        waveform.shape[0],
    )

    suffix = f"_encodec_{args.quality}"
    output_path = input_path.with_name(
        f"{input_path.stem}{suffix}.{args.output_format}"
    )

    if args.output_format == "wav":
        torchaudio.save(str(output_path), decoded, sample_rate)
    else:
        torchaudio.save(str(output_path), decoded, sample_rate, format="mp3")

    print(f"Saved decoded audio to {output_path}")


if __name__ == "__main__":
    main()
