
import tarfile
import zipfile
from pathlib import Path

__all__ = ["ensure_dataset", "DatasetDownloadError"]


class DatasetDownloadError(Exception):
    """Raised when downloading or extracting a dataset fails."""


def _download_file(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest}")
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


def _extract_zip(zip_path: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest)
    except Exception as e:
        raise DatasetDownloadError(f"Failed extracting {zip_path}") from e


def _download_nsynth(dataset_dir: Path) -> None:
    for subset in ["train", "valid", "test"]:
        url = f"http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{subset}.jsonwav.tar.gz"
        tar_path = dataset_dir / f"nsynth-{subset}.jsonwav.tar.gz"
        _download_file(url, tar_path)
        _extract_tar(tar_path, dataset_dir)
        tar_path.unlink(missing_ok=True)
        orig_path = dataset_dir / f"nsynth-{subset}"
        simple_path = dataset_dir / subset
        orig_path.rename(simple_path)


def _download_idmt(dataset_dir: Path) -> None:
    url = "https://zenodo.org/records/7544164/files/IDMT-SMT-DRUMS-V2.zip"
    tar_path = dataset_dir / "idmt-smt.zip"
    _download_file(url, tar_path)
    _extract_zip(tar_path, dataset_dir)
    tar_path.unlink(missing_ok=True)


def ensure_dataset(name: str, root: Path) -> Path:
    if name not in {"nsynth", "idmt_smt"}:
        raise ValueError("name must be 'nsynth' or 'idmt_smt'")

    dataset_dir = root / name

    if dataset_dir.exists() and any(dataset_dir.rglob("*.wav")):
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if name == "nsynth":
        _download_nsynth(dataset_dir)
    else:
        _download_idmt(dataset_dir)

    return dataset_dir
