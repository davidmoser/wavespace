import json
import pathlib
import subprocess


def extract_tracks(path: str):
    src = pathlib.Path(path)  # e.g. python split_tracks.py input.mp4
    probe = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "a",  # audio streams only
        "-show_entries", "stream=index",
        "-of", "json", str(src)])
    for s in json.loads(probe)["streams"]:
        i = s["index"]  # stream index inside the file
        out = src.with_suffix(f".track{i}.m4a")
        subprocess.check_call([
            "ffmpeg", "-y",  # overwrite if exists
            "-i", str(src),
            "-map", f"0:a:{i}",  # pick one audio stream
            "-c", "copy",  # no re-encoding
            str(out)])
        print("wrote", out)
