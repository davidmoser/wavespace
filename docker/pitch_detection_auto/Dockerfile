FROM gebregl/wavespace-dependencies:v2
WORKDIR /

COPY /docker/pitch_detection_auto/rp_handler.py /rp_handler.py
COPY /docker/pitch_detection_auto/sweep_run.py /sweep_run.py
COPY /docker/pitch_detection_auto/single_run.py /single_run.py
COPY /pitch_detection_auto /pitch_detection_auto/
COPY /datasets /datasets/
COPY /utils /utils/

COPY /docker/pitch_detection_auto/test_input.json /test_input.json

CMD ["python3", "-u", "rp_handler.py"]