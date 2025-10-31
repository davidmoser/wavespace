import sys

from pitch_detection_supervised.train import single_run_resume


def main():
    run_id = sys.argv[1]
    single_run_resume(run_id)


if __name__ == "__main__":
    main()
