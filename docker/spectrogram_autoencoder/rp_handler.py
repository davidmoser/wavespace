import runpod

from spectrogram_autoencoder.configuration import Configuration
from spectrogram_autoencoder.train import train


def handler(event):
    print(f"Worker Start")
    input = event['input']
    cfg = Configuration(**input)

    print(f"Training with configuration: {cfg}")

    train(cfg)

    return "Training finished"


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
