import json
import sys

from pitch_detection_supervised.configuration import Configuration
from pitch_detection_supervised.train import single_run


def main():
    cfg_data = json.loads(sys.argv[1])
    cfg = Configuration(**cfg_data)
    single_run(cfg)


if __name__ == "__main__":
    main()
