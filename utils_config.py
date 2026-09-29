import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
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

def load_config(config_path):
    """
    Load a yaml config file and convert it to a SimpleNamespace.
    
    Arguments:
    ----------
        config_path: path to the yaml config file
    Returns:
    --------
        SimpleNamespace: namespace
    """
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

    