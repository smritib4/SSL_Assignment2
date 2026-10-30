"""Dataset construction and loaders.

The stratified 80/20 split is carried from Assignment 1. With about 150 images
per class, an unstratified split moves several images between classes and a
1% accuracy gap is only five validation images. Every experiment uses seed 0
and val_fraction 0.20 so the 1,920 / 480 cut is the same one.

Pretraining uses that same training cut. The validation images are not fed to
SimCLR: an encoder that has already seen the validation pixels (even without
labels) makes the validation accuracy look better than it is.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform
from .utils import seed_worker

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def natural_key(name: str) -> list[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


class LabeledImages(Dataset):
    """Image list with one transform, or two independent draws for SimCLR."""

    def __init__(self, samples: list[tuple[str, int]], transform, two_view: bool) -> None:
        self.samples = samples
        self.transform = transform
        self.two_view = two_view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.two_view:
            # One decode, two stochastic transforms: a positive pair.
            return self.transform(image), self.transform(image), label
        return self.transform(image), label


class FlatImageDataset(Dataset):
    """A directory of unlabelled images. Returns `(tensor, filename)`."""

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Not a directory: {self.root}")
        self.paths = sorted(
            (p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS),
            key=lambda p: natural_key(p.name),
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


def stratified_split(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for class_id in np.unique(targets_arr):
        class_indices = np.flatnonzero(targets_arr == class_id)
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_fraction))
        n_val = max(1, min(n_val, len(class_indices) - 1))
        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_datasets(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    train_root = root / data_cfg["train_dir"]
    test_root = root / data_cfg["test_dir"]
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_root.resolve()}\n"
            "Run `python scripts/prepare_data.py --zip <downloaded.zip>` first."
        )

    folder = datasets.ImageFolder(train_root)
    if data_cfg.get("split", "stratified") != "stratified":
        raise ValueError("Only split=stratified is supported. An unstratified split is not comparable.")
    train_idx, val_idx = stratified_split(
        list(folder.targets), float(data_cfg["val_fraction"]), int(cfg["experiment"]["seed"])
    )
    if set(train_idx) & set(val_idx):
        raise RuntimeError("Train/val split leaked indices")

    mode = str(cfg["experiment"]["mode"]).lower()
    two_view = mode in {"pretrain", "joint"}
    train_tf = build_train_transform(data_cfg, cfg["augment"])
    eval_tf = build_eval_transform(data_cfg)

    train_samples = [folder.samples[i] for i in train_idx]
    val_samples = [folder.samples[i] for i in val_idx]
    train_dataset = LabeledImages(train_samples, train_tf, two_view=two_view)
    val_dataset = LabeledImages(val_samples, eval_tf, two_view=False)

    test_dataset = None
    if test_root.is_dir() and any(test_root.iterdir()):
        # ImageFolder only if class subdirectories exist. A flat test folder is
        # handled by predict.py instead.
        class_dirs = [p for p in test_root.iterdir() if p.is_dir()]
        if class_dirs:
            test_folder = datasets.ImageFolder(test_root)
            if test_folder.classes != folder.classes:
                raise ValueError(
                    "Test classes do not match train classes.\n"
                    f"train: {folder.classes}\ntest:  {test_folder.classes}"
                )
            test_dataset = LabeledImages(list(test_folder.samples), eval_tf, two_view=False)

    train_targets = [folder.targets[i] for i in train_idx]
    val_targets = [folder.targets[i] for i in val_idx]
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": list(folder.classes),
        "train_targets": train_targets,
        "val_targets": val_targets,
        "two_view": two_view,
        "eval_transform": eval_tf,
    }


def build_loaders(cfg: dict[str, Any], bundle: dict[str, Any]) -> dict[str, DataLoader | None]:
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    available = os.cpu_count() or 1
    num_workers = min(int(data_cfg["num_workers"]), available)
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"]["seed"]))
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    # Contrastive loss and batch-norm both want a full batch. The supervised
    # baseline keeps the tail so no labelled image is dropped.
    drop_last = bool(bundle["two_view"])
    loaders: dict[str, DataLoader | None] = {
        "train": DataLoader(
            bundle["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
            **common,
        ),
        "val": DataLoader(bundle["val"], batch_size=batch_size, shuffle=False, **common),
        "test": None,
    }
    if bundle["test"] is not None:
        loaders["test"] = DataLoader(bundle["test"], batch_size=batch_size, shuffle=False, **common)
    return loaders


def describe_dataset(bundle: dict[str, Any]) -> str:
    lines = [
        f"Classes ({len(bundle['class_names'])}): {bundle['class_names']}",
        f"Train images: {len(bundle['train'])}",
        f"Val images:   {len(bundle['val'])}",
        f"Two-view batches: {bundle['two_view']}",
    ]
    if bundle["test"] is not None:
        lines.append(f"Test images:  {len(bundle['test'])}")
    counts = Counter(bundle["train_targets"])
    per_class = [counts.get(i, 0) for i in range(len(bundle["class_names"]))]
    lines.append(f"Train images per class: min={min(per_class)} max={max(per_class)}")
    return "\n".join(lines)
