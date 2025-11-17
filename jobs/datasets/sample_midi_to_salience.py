from datasets.midi_to_salience import midi_to_salience, prepare_cqts
from utils.plot_cqt_comparison import plot_cqt_comparison


def sample_midi_to_salience(
        midi_path: str,
        wav_path: str,
        png_path: str,
        label_type: str,
):
    cqts = prepare_cqts(
        audio_path=wav_path,
        chunk_duration=30.0,
        frame_rate=75,
    )
    salience = midi_to_salience(
        midi_path=midi_path,
        chunk_duration=30.0,
        frame_rate=75,
        label_type=label_type,
        cqts=cqts,
    )
    plot_cqt_comparison(
        audio_file=wav_path,
        prediction=salience[0].transpose(0, 1),
        output_image=png_path,
        duration_seconds=30,
        scale_values=0.5
        # prediction_vertical_pixels: int = 128,
        # cqt_bins: int = 84,
        # cqt_bins_per_octave: int = 12,
        # cqt_hop_length: int = 256,
        # cqt_fmin: float = 32.7,
        # cqt_vertical_pixels: int = 128,
    )


if __name__ == "__main__":
    label_type = "power"
    sample_midi_to_salience(
        midi_path="../../resources/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi",
        wav_path="../../resources/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav",
        png_path=f"../../resources/maestro-v3.0.0/samples/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_{label_type}.png",
        label_type=label_type
    )
