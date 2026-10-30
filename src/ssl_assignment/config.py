"""YAML experiment configs merged onto defaults, then CLI overrides.

Every config file only states what differs from the defaults. `--set a.b=value`
is applied last so a one-factor ablation does not need its own file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "unnamed",
        "notes": "",
        "seed": 0,
        # supervised | pretrain | joint
        # Fine-tuning and linear probing are supervised runs that set
        # ssl.init_checkpoint (and model.freeze_encoder for the probe).
        "mode": "supervised",
    },
    "data": {
        "root": "data",
        "train_dir": "train",
        "test_dir": "test",
        "val_fraction": 0.20,
        "split": "stratified",
        "batch_size": 64,
        "num_workers": 2,
        "img_size": 128,
        # Normalization constants only. These are not pretrained weights.
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "augment": {
        "train_crop_scale": [0.65, 1.0],
        "hflip": 0.5,
        "color_jitter": 0.2,
        "jitter_hue": 0.05,
        "jitter_p": 1.0,
        "grayscale_p": 0.0,
        "blur_p": 0.0,
        "blur_sigma": [0.1, 2.0],
        "random_erasing": 0.1,
    },
    "model": {
        "name": "resnet18",
        "dropout": 0.0,
        "freeze_encoder": False,
    },
    "ssl": {
        "temperature": 0.2,
        "proj_hidden": 512,
        "proj_dim": 128,
        "lambda_contrast": 0.2,
        "init_checkpoint": "",
    },
    "optim": {
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "head_lr_mult": 1.0,
        "weight_decay": 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "min_lr": 1.0e-6,
        "epochs": 50,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "amp": True,
    },
    "eval": {
        "tta": "none",
    },
    "output": {
        "root": "runs",
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `updates` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `a.b.c=value` strings. The key must already exist."""
    if not overrides:
        return cfg
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section '{key}' in override '{item}'")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(
                f"Unknown config key '{dotted}'. Add it to DEFAULTS before overriding it."
            )
        node[keys[-1]] = _coerce(raw_value)
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            file_cfg = yaml.safe_load(handle) or {}
        cfg = deep_update(cfg, file_cfg)
        cfg["experiment"]["config_path"] = str(path)
        if cfg["experiment"]["name"] == "unnamed":
            cfg["experiment"]["name"] = Path(path).stem
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
