import librosa

from utils.stft_video import create_stft_video


create_stft_video(
    audio_file="../../resources/Gentle on My Mind - Cotton Pickin Kids/Gentle on My Mind - Cotton Pickin Kids.mp3",
    output_video="../../resources/Gentle on My Mind - Cotton Pickin Kids/all_stft_512.mp4",
    duration_seconds=20,
    scale_values=1,
    stft_bins=512,
    stft_bins_per_column=24,
    stft_frames_per_s=20,
    stft_fmin=float(librosa.midi_to_hz(0)),
    stft_vertical_pixels=256,
    column_width_pixels=40,
    normalization="global",
    colormap="viridis",
)
