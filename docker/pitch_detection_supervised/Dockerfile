FROM gebregl/wavespace-dependencies:v2
WORKDIR /

COPY /docker/pitch_detection_supervised/rp_handler.py /rp_handler.py
COPY /docker/pitch_detection_supervised/sweep_run.py /sweep_run.py
COPY /docker/pitch_detection_supervised/single_run.py /single_run.py
COPY /pitch_detection_supervised /pitch_detection_supervised/
COPY /datasets /datasets/

COPY /docker/pitch_detection_supervised/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]
