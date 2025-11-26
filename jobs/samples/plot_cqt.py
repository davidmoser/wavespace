import librosa

from utils.plot_cqt_comparison import plot_cqt

plot_cqt(
    audio_file="../../resources/Phases - TWO LANES/Phases - TWO LANES.mp3",
    output_image="../../resources/Phases - TWO LANES/all_cqt_256_nolog.png",
    duration_seconds=None,
    # scale_values=scale_values,
    cqt_bins=256,
    cqt_bins_per_octave=36,
    cqt_frames_per_s=20,
    cqt_fmin=float(librosa.midi_to_hz(18)),
    cqt_vertical_pixels=512,
    pixels_per_second=60,
    log_values=False,
)

