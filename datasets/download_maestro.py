#!/usr/bin/env python3
# Downloads MAESTRO v3.0.0 and unzips to ../resources/maestro

from __future__ import annotations
import sys, os, zipfile, urllib.request
from pathlib import Path

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
ZIP_NAME = "maestro-v3.0.0.zip"

def progress(blocks: int, block_size: int, total_size: int):
    if total_size <= 0:
        return
    downloaded = blocks * block_size
    pct = min(100, int(downloaded * 100 / total_size))
    bar = "#" * (pct // 2)
    sys.stdout.write(f"\rDownloading: [{bar:<50}] {pct:3d}%")
    sys.stdout.flush()

def download_maestro(target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / ZIP_NAME

    if not zip_path.exists():
        print(f"Downloading MAESTRO v3.0.0 to {zip_path} ...")
        try:
            urllib.request.urlretrieve(MAESTRO_URL, zip_path, reporthook=progress)
            sys.stdout.write("\n")
        except Exception as e:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            sys.stdout.write("\n")
            raise SystemExit(f"Download failed: {e}")
    else:
        print(f"Archive already exists: {zip_path}")

    # Unzip
    print(f"Unzipping into {target_dir} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only if not already extracted
            marker = target_dir / ".unzipped_v3.0.0"
            if marker.exists():
                print("Detected previous extraction. Skipping unzip.")
            else:
                zf.extractall(target_dir)
                marker.touch()
    except zipfile.BadZipFile:
        raise SystemExit("Archive is corrupt. Delete it and run this script again.")

    # Done
    print("Done.")

