import librosa

from utils.cqt_video import create_cqt_video


create_cqt_video(
    audio_file="../../resources/Gentle on My Mind - Cotton Pickin Kids/Gentle on My Mind - Cotton Pickin Kids.mp3",
    output_video="../../resources/Gentle on My Mind - Cotton Pickin Kids/all_cqt_512.mp4",
    duration_seconds=20,
    scale_values=1,
    cqt_bins=512,
    cqt_bins_per_octave=24,
    cqt_frames_per_s=20,
    cqt_fmin=float(librosa.midi_to_hz(0)),
    cqt_vertical_pixels=256,
    column_width_pixels=40,
    normalization="global",
    colormap="viridis",
)
