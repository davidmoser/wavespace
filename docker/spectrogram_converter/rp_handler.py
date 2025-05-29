import runpod

from spectrogram_converter.configuration import Configuration
from spectrogram_converter.convert import convert


def handler(event):
    print(f"Worker Start")
    input = event['input']
    cfg = Configuration(**input)

    print(f"Converting with configuration: {cfg}")

    convert(cfg)

    return "Converting finished"


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
