import json
import os
import shutil
import subprocess
from typing import Any

def run_root(cfg: Any) -> str:
    output_dir = getattr(cfg.run, "output_dir", "output")
    return os.path.join(output_dir, cfg.run.name)

def seed_dir(cfg: Any) -> str:
    return os.path.join(run_root(cfg), f"s{cfg.run.seed}")

def init_run(cfg: Any, config_path: str) -> str:
    """
    Create output/<run.name>/s<seed>/ and keep a copy of the config in it.
    """
    d = seed_dir(cfg)
    os.makedirs(d, exist_ok=True)
    shutil.copy(config_path, os.path.join(d, "config.yaml"))
    return d

def alpha_dir(base_dir, alpha: float, multi_val: bool=False) -> str:
    """Sub-directory per alpha only when `alpha_list` has multiple values."""
    d = os.path.join(base_dir, f"a{alpha:.2f}") if multi_val else base_dir
    os.makedirs(d, exist_ok=True)
    return d

def write_meta(out_dir: str, **fields: Any) -> None:
    """Store run info (seed, best epoch, best val EER, ...) in meta.json inside out_dir"""
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(fields, f, indent=2)
