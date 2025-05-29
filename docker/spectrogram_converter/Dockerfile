FROM gebregl/wavespace-dependencies:v1
WORKDIR /

COPY /docker/spectrogram_converter/rp_handler.py /rp_handler.py
COPY /spectrogram_converter /spectrogram_converter

COPY /docker/spectrogram_converter/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]