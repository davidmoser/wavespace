import librosa

from utils.stft_video import create_stft_video


create_stft_video(
    audio_file="../../resources/maestro-v3.0.0/2006/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.wav",
    output_video="../../resources/maestro-v3.0.0/samples/stft_video.mp4",
    duration_seconds=60,
    scale_values=1,
    stft_bins=2048,
    stft_bins_per_column=32,
    stft_frames_per_s=10,
    stft_fmin=float(librosa.midi_to_hz(0)),
    stft_vertical_pixels=256,
    column_width_pixels=40,
    normalization="global",
    colormap="viridis",
)
