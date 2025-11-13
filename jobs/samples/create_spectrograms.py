# wav_to_spectrogram_image(
#     in_wav="../resources/Gentle on My Mind - Cotton Pickin Kids/other.wav",
#     out_img="../resources/other_stft_512.png",
#     n_fft=512,
#     hop_ratio=0.1,
# )
#

# for i in range(5):
#     path = f"../resources/musdb18/test/Al James - Schoolboy Facination.stem.track{i}.m4a"
#     wav_to_spectrogram_image(
#         in_wav=path,
#         out_img=f"../resources/aljames_track{i}_stft_4196.png",
#         n_fft=4196,
#         hop_ratio=0.1,
#     )


# wav_to_spectrogram_image(
#     in_wav="../resources/Gentle on My Mind - Cotton Pickin Kids/other.wav",
#     out_img="../resources/other_stft_4196.png",
#     n_fft=4196,
#     hop_ratio=0.1,
# )
#
# wav_to_spectrogram_image(
#     in_wav="../resources/Gentle on My Mind - Cotton Pickin Kids/vocals.wav",
#     out_img="../resources/vocals_stft_4196.png",
#     n_fft=4196,
#     hop_ratio=0.1,
# )

# wav_to_log_spectrogram_image(
#     in_wav="../resources/Gentle on My Mind - Cotton Pickin Kids/other.wav",
#     out_img="../resources/Gentle on My Mind - Cotton Pickin Kids/other_logspc_4096.png",
#     sr=22_000,
#     n_fft=4096,
#     log_bins=256,
#     hop=441,
# )