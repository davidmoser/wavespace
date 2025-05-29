import re
from spectrogram_autoencoder.spec_auto_v1 import SpecAutoNet as NetV1
from spectrogram_autoencoder.spec_auto_v2 import SpecAutoNet as NetV2
from spectrogram_autoencoder.spec_auto_v3 import SpecAutoNet as NetV3
from spectrogram_autoencoder.spec_auto_v4 import SpecAutoNet as NetV4

_MODEL_REGISTRY = {1: NetV1, 2: NetV2, 3: NetV3, 4: NetV4}


def get_model(version: str):
    v_num = int(re.search(r"v(\d+)_?", version).group(1))
    try:
        return _MODEL_REGISTRY[v_num]()
    except KeyError as e:
        raise ValueError(f"Unknown model version {version}") from e