import librosa

from utils.plot_cqt_comparison import plot_cqt

plot_cqt(
    audio_file="../../resources/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav",
    output_image="../../resources/maestro-v3.0.0/samples/MIDI-Unprocessed_SMF_02_Track05_wav_cqt_high.png",
    duration_seconds=20,
    # scale_values=scale_values,
    cqt_bins=256,
    cqt_bins_per_octave=24,
    cqt_frames_per_s=20,
    cqt_fmin=float(librosa.midi_to_hz(0)),
    cqt_vertical_pixels=256,
    pixels_per_second=50,
)

