import librosa

from utils.cqt_video import create_cqt_video


create_cqt_video(
    audio_file="../../resources/maestro-v3.0.0/2006/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.wav",
    output_video="../../resources/maestro-v3.0.0/samples/cqt_video.mp4",
    duration_seconds=60,
    scale_values=1,
    cqt_bins=256,
    cqt_bins_per_octave=24,
    cqt_frames_per_s=20,
    cqt_fmin=float(librosa.midi_to_hz(0)),
    cqt_vertical_pixels=256,
    column_width_pixels=40,
    normalization="global",
    colormap="viridis",
)
