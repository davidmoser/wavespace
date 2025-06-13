FROM gebregl/wavespace-dependencies:v2
WORKDIR /

COPY /docker/pitch_channel_eval/rp_handler.py /rp_handler.py
COPY /pitch_detection /pitch_detection/

COPY /docker/pitch_channel_eval/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]
