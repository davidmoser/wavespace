FROM gebregl/wavespace-dependencies:v2
WORKDIR /

COPY /docker/pitch_detection/rp_handler.py /rp_handler.py
COPY /docker/pitch_detection/sweep_run.py /sweep_run.py
COPY /pitch_detection /pitch_detection/

COPY /docker/pitch_detection/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]