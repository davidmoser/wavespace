import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from unmixer.dataset_setup import ensure_dataset
from unmixer import mix_creator


def create_dummy_wav(path: Path, sr: int = 16000, dur: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(int(sr * dur))
    sf.write(path, data, sr)


def test_ensure_dataset_existing(tmp_path: Path) -> None:
    ds_root = tmp_path / "nsynth"
    raw_dir = ds_root / "raw"
    audio_dir = ds_root / "audio_sr16000"
    dummy = raw_dir / "sound.wav"
    create_dummy_wav(dummy)
    (audio_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(dummy, audio_dir / "sound.wav")

    out = ensure_dataset("nsynth", tmp_path, target_sr=16000, mono=True)
    assert out.exists()
    assert list(out.rglob("*.wav"))


def test_generate_mixes_single(tmp_path: Path) -> None:
    # prepare minimal datasets
    nsynth = tmp_path / "nsynth" / "audio_sr16000"
    idmt = tmp_path / "idmt_smt" / "audio_sr16000"
    create_dummy_wav(nsynth / "a.wav")
    create_dummy_wav(idmt / "b.wav")

    out_root = tmp_path / "mixes"

    class DummyScaper:
        def __init__(self, duration, sr, room_cfg):
            self.duration = duration
            self.sr = sr
            self.room_cfg = room_cfg
            self.calls = []

        def add_source(self, filepath, source_time, room_position, allow_repitch):
            self.calls.append((filepath, source_time))

        def generate(self, out_audio, out_metadata, convolve=True):
            sf.write(out_audio, np.zeros(int(self.sr * self.duration)), self.sr)
            with open(out_metadata, "w") as f:
                json.dump({"dummy": True}, f)

    with mock.patch.dict("sys.modules", {"spatialscaper": mock.MagicMock(SpatialScaper=DummyScaper)}):
        mix_creator.generate_mixes(
            {"nsynth": nsynth, "idmt_smt": idmt},
            out_root,
            num_mixes=1,
            max_events=3,
            mix_length_s=2.0,
            target_sr=16000,
            random_seed=0,
        )

    wavs = list(out_root.glob("*.wav"))
    metas = list(out_root.glob("*.json"))
    assert len(wavs) == 1 and len(metas) == 1
    with open(metas[0]) as f:
        json.load(f)
