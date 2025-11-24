import librosa

from utils.plot_cqt_comparison import plot_cqt

plot_cqt(
    audio_file="../../resources/Gentle on My Mind - Cotton Pickin Kids/Gentle on My Mind - Cotton Pickin Kids.mp3",
    output_image="../../resources/Gentle on My Mind - Cotton Pickin Kids/all_cqt_512.png",
    duration_seconds=20,
    # scale_values=scale_values,
    cqt_bins=512,
    cqt_bins_per_octave=24,
    cqt_frames_per_s=20,
    cqt_fmin=float(librosa.midi_to_hz(0)),
    cqt_vertical_pixels=256,
    pixels_per_second=50,
)

