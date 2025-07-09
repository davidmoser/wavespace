from __future__ import annotations

import tarfile
from pathlib import Path

__all__ = ["ensure_dataset", "DatasetDownloadError"]



class DatasetDownloadError(Exception):
    """Raised when downloading or extracting a dataset fails."""


def _download_file(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        raise DatasetDownloadError(f"Failed downloading {url}") from e


def _extract_tar(tar_path: Path, dest: Path) -> None:
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(dest)
    except Exception as e:
        raise DatasetDownloadError(f"Failed extracting {tar_path}") from e


def _download_nsynth(dataset_dir: Path) -> None:
    url = "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-full.tar.gz"
    tar_path = dataset_dir / "nsynth-full.tar.gz"
    _download_file(url, tar_path)
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _extract_tar(tar_path, raw_dir)
    tar_path.unlink(missing_ok=True)


def _download_idmt(dataset_dir: Path) -> None:
    url = "https://zenodo.org/record/7544164/files/IDMT-SMT-Drums.tar.gz"
    tar_path = dataset_dir / "idmt-smt.tar.gz"
    _download_file(url, tar_path)
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _extract_tar(tar_path, raw_dir)
    tar_path.unlink(missing_ok=True)


def _convert_audio(raw_dir: Path, audio_dir: Path, target_sr: int, mono: bool) -> None:
    import librosa
    import soundfile as sf

    for wav in raw_dir.rglob("*.wav"):
        y, _ = librosa.load(wav, sr=target_sr, mono=mono)
        out_path = audio_dir / wav.relative_to(raw_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y, target_sr)


def ensure_dataset(name: str, root: Path, target_sr: int, mono: bool) -> Path:
    if name not in {"nsynth", "idmt_smt"}:
        raise ValueError("name must be 'nsynth' or 'idmt_smt'")

    dataset_dir = root / name
    audio_dir = dataset_dir / f"audio_sr{target_sr}"

    if audio_dir.exists() and any(audio_dir.rglob("*.wav")):
        return audio_dir

    raw_dir = dataset_dir / "raw"
    if not raw_dir.exists() or not any(raw_dir.rglob("*.wav")):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if name == "nsynth":
            _download_nsynth(dataset_dir)
        else:
            _download_idmt(dataset_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    _convert_audio(raw_dir, audio_dir, target_sr, mono)
    return audio_dir
