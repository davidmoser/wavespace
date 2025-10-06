"""Utilities shared by RunPod entrypoints for supervised pitch detection."""
from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from pitch_detection_supervised.configuration import Config
from pitch_detection_supervised.dilated_tcn import DilatedTCN
from pitch_detection_supervised.make_loaders import make_loaders


Factory = Callable[[Config], Any]


def _import_attr(target: str) -> Callable[..., Any]:
    module_name, _, attribute_name = target.rpartition('.')
    if not module_name or not attribute_name:
        raise ValueError(f"Invalid import target '{target}'. Expected format 'module.attr'.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Attribute '{attribute_name}' not found in module '{module_name}'.") from exc


def _call_factory(factory: Callable[..., Any], config: Config, params: Dict[str, Any]) -> Any:
    """Call *factory* passing ``config`` when supported."""

    try:
        return factory(config=config, **params)
    except TypeError as exc:
        first_error = exc
    else:  # pragma: no cover - sanity guard
        raise AssertionError('factory invocation unexpectedly succeeded twice')

    try:
        return factory(config, **params)
    except TypeError:
        pass

    try:
        return factory(**params)
    except TypeError as exc:
        message = (
            f"Unable to call factory '{factory}' with provided parameters. "
            f"Last error: {exc} (first error: {first_error})"
        )
        raise TypeError(message) from exc


def build_config(payload: Dict[str, Any]) -> Config:
    """Construct a :class:`Config` from the request payload."""

    config_dict = payload.get('config')
    if config_dict is None:
        config_dict = payload
    if not isinstance(config_dict, dict):
        raise TypeError("'config' must be a dictionary when provided.")
    return Config.from_dict(config_dict)


def instantiate_model(spec: Optional[Any], config: Config) -> Module:
    """Create the model described by *spec* or a default ``DilatedTCN``."""

    if spec is None:
        return DilatedTCN(
            n_classes=config.n_classes,
            seq_len=config.seq_len,
            latent_dim=config.latent_dim,
        )

    target: Optional[str]
    params: Dict[str, Any]
    if isinstance(spec, str):
        target = spec
        params = {}
    elif isinstance(spec, dict):
        target = spec.get('target')
        if not isinstance(target, str):
            raise ValueError("Model specification dictionary must contain a 'target' string.")
        params = dict(spec.get('params', {}))
    else:
        raise TypeError('Model specification must be a string, dict, or null.')

    factory = _import_attr(target)
    result = _call_factory(factory, config, params)
    if not isinstance(result, Module):
        raise TypeError('Model factory must return an instance of torch.nn.Module.')
    return result


def load_dataset_from_path(path_str: str) -> Any:
    """Load a dataset iterable from *path_str* supporting ``.pt``/``.pth`` and JSON files."""

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path '{path}' does not exist.")

    suffix = path.suffix.lower()
    if suffix in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)
    if suffix in {'.jsonl', '.ndjson'}:
        with path.open('r', encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]

    raise ValueError(f"Unsupported dataset file extension '{suffix}' for path '{path}'.")


def _resolve_dataset(payload: Dict[str, Any], prefix: str) -> Optional[Any]:
    value = payload.get(prefix)
    if value is None:
        path_key = f'{prefix}_path'
        value = payload.get(path_key)
    if value is None:
        return None

    if isinstance(value, str):
        return load_dataset_from_path(value)
    if isinstance(value, dict):
        if 'path' in value:
            return load_dataset_from_path(value['path'])
        if 'data_path' in value:
            return load_dataset_from_path(value['data_path'])
        if 'data' in value:
            return value['data']
    raise TypeError(f"Unsupported dataset specification for '{prefix}'.")


def _instantiate_loader(spec: Optional[Any], config: Config) -> Optional[DataLoader]:
    if spec is None:
        return None

    target: Optional[str]
    params: Dict[str, Any]
    if isinstance(spec, str):
        target = spec
        params = {}
    elif isinstance(spec, dict):
        target = spec.get('target')
        if not isinstance(target, str):
            raise ValueError("Loader specification dictionary must contain a 'target' string.")
        params = dict(spec.get('params', {}))
    else:
        raise TypeError('Loader specification must be a string, dict, or null.')

    factory = _import_attr(target)
    result = _call_factory(factory, config, params)
    if result is None:
        return None
    if isinstance(result, DataLoader):
        return result
    raise TypeError('Loader factory must return a torch.utils.data.DataLoader or None.')


def prepare_loaders(payload: Dict[str, Any], config: Config) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation loaders from the payload."""

    train_loader = _instantiate_loader(payload.get('train_loader'), config)
    val_loader = _instantiate_loader(payload.get('val_loader'), config)

    if train_loader is not None:
        return train_loader, val_loader

    train_data = _resolve_dataset(payload, 'train_data')
    if train_data is None:
        raise ValueError("Request must provide either 'train_loader' or 'train_data'.")

    val_data = _resolve_dataset(payload, 'val_data')
    loader_config = payload.get('loader_config')
    if loader_config is not None:
        if not isinstance(loader_config, dict):
            raise TypeError("'loader_config' must be a dictionary when provided.")
        config = replace(config, **{k: v for k, v in loader_config.items() if hasattr(config, k)})

    train_loader, derived_val_loader = make_loaders(train_data, val_data, config)
    if train_loader is None:
        raise RuntimeError('Loader creation returned None for training data.')
    if val_loader is None:
        val_loader = derived_val_loader
    return train_loader, val_loader
