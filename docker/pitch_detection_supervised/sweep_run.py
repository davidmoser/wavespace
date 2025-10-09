import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from pitch_detection_supervised.configuration import Config
from pitch_detection_supervised.make_loaders import make_loaders
from pitch_detection_supervised.train import sweep_run as run_sweep

from run_utils import instantiate_model, load_dataset_from_path


@lru_cache(maxsize=None)
def _load_dataset(path: str) -> Any:
    return load_dataset_from_path(path)


def _train_loader_factory(config: Config):
    train_path = os.environ.get('PITCH_SUP_TRAIN_DATA_PATH')
    if not train_path:
        raise RuntimeError('PITCH_SUP_TRAIN_DATA_PATH environment variable must be set.')
    train_data = _load_dataset(train_path)
    val_path = os.environ.get('PITCH_SUP_VAL_DATA_PATH')
    val_data = _load_dataset(val_path) if val_path else None
    train_loader, _ = make_loaders(train_data, val_data, config)
    if train_loader is None:
        raise RuntimeError('make_loaders returned None for training data.')
    return train_loader


def _val_loader_factory(config: Config):
    val_path = os.environ.get('PITCH_SUP_VAL_DATA_PATH')
    if not val_path:
        return None
    val_data = _load_dataset(val_path)
    _, val_loader = make_loaders(None, val_data, config)
    return val_loader


def _model_factory(config: Config):
    spec_json = os.environ.get('PITCH_SUP_MODEL_SPEC')
    spec: Optional[Any]
    if spec_json:
        loaded = json.loads(spec_json)
        if not isinstance(loaded, (dict, str)) and loaded is not None:
            raise TypeError('PITCH_SUP_MODEL_SPEC must encode a string or dictionary.')
        spec = loaded
    else:
        spec = None
    return instantiate_model(spec, config)


def _base_config() -> Optional[Config]:
    base_json = os.environ.get('PITCH_SUP_BASE_CONFIG')
    if not base_json:
        return None
    data = json.loads(base_json)
    if not isinstance(data, dict):
        raise TypeError('PITCH_SUP_BASE_CONFIG must encode a dictionary.')
    return Config(**data)


def main() -> None:
    base_config = _base_config()
    val_factory = _val_loader_factory if os.environ.get('PITCH_SUP_VAL_DATA_PATH') else None
    project = os.environ.get('PITCH_SUP_PROJECT', 'pitch-detection-supervised')
    run_sweep(
        model_factory=_model_factory,
        train_loader_factory=_train_loader_factory,
        val_loader_factory=val_factory,
        base_config=base_config,
        project=project,
    )


if __name__ == '__main__':
    main()
