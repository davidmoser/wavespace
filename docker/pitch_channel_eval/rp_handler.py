import runpod

from pitch_detection_auto.channel_eval import ChannelEvalConfig, evaluate_channels


def handler(event):
    print("Worker Start")
    cfg = ChannelEvalConfig.from_dict(event["input"])
    evaluate_channels(cfg)
    return "Evaluation finished"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
