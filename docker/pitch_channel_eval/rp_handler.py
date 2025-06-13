import runpod

from pitch_detection import ChannelEvalConfig, evaluate_channels


def handler(event):
    print("Worker Start")
    cfg = ChannelEvalConfig(**event["input"])
    evaluate_channels(cfg)
    return "Evaluation finished"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
