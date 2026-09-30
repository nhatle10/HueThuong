from types import SimpleNamespace
from typing import Any
import yaml


def dict_to_namespace(d: Any) -> Any:
    """
    Recursively converts a dictionary to SimpleNamespace for dot notation.

    Arguments:
    ----------
        d: dict or list or primitive to convert
    Returns:
    --------
        SimpleNamespace or list or primitive
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


def load_config(config_path: str) -> SimpleNamespace:
    """
    Load a yaml config file and convert it to a SimpleNamespace.

    Arguments:
    ----------
        config_path: path to the yaml config file
    Returns:
    --------
        SimpleNamespace: namespace
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg_dict)
    if not isinstance(cfg, SimpleNamespace):
        raise TypeError(f"Expected root of YAML configuration to be a mapping, got {type(cfg).__name__}")
    return cfg


    