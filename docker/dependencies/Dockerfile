FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
WORKDIR /

RUN apt-get update && apt-get install -y libsndfile1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt