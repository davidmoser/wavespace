FROM gebregl/wavespace-dependencies:v2
WORKDIR /

COPY /docker/spectrogram_autoencoder/rp_handler.py /rp_handler.py
COPY /docker/spectrogram_autoencoder/sweep_run.py /sweep_run.py
COPY /spectrogram_autoencoder /spectrogram_autoencoder/

COPY /docker/spectrogram_autoencoder/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]