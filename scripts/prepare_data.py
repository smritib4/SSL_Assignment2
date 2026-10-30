"""Unpack the dataset into the layout training expects.

    python scripts/prepare_data.py --zip C:/Users/you/Downloads/dataset.zip
    python scripts/prepare_data.py --src C:/Users/you/Downloads/extracted_folder

The Drive link is the same one as Assignment 1 (the PDF only shows the words
"Download Dataset"):

    https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_ALIASES = {"train", "training", "train_set"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS and not path.name.startswith(".")


def is_class_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    return any(is_image(child) for child in path.iterdir() if child.is_file())


def is_split_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    return sum(1 for child in path.iterdir() if is_class_dir(child)) >= 2


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            m
            for m in archive.namelist()
            if not m.startswith("__MACOSX") and "/._" not in m and not m.split("/")[-1].startswith("._")
        ]
        archive.extractall(dest, members=members)
    print(f"Extracted {len(members)} entries to {dest}")
    return dest


def find_split_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if not is_split_dir(candidate):
            continue
        name = candidate.name.lower()
        key = "train" if name in TRAIN_ALIASES else name
        found.setdefault(key, candidate)
    return found


def copy_split(src: Path, dest: Path) -> Counter:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        target_dir = dest / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image in sorted(class_dir.rglob("*")):
            if is_image(image):
                shutil.copy2(image, target_dir / image.name)
                counts[class_dir.name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=str)
    group.add_argument("--src", type=str)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    staging: Path | None = None
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            sys.exit(f"Zip not found: {zip_path}")
        staging = Path("data_raw")
        if staging.exists():
            shutil.rmtree(staging)
        search_root = extract_zip(zip_path, staging)
    else:
        search_root = Path(args.src)
        if not search_root.is_dir():
            sys.exit(f"Source folder not found: {search_root}")

    splits = find_split_dirs(search_root)
    if "train" not in splits:
        sys.exit(
            f"Could not find a train directory with class subfolders under {search_root}.\n"
            f"Splits detected: {sorted(splits) or 'none'}"
        )
    for name in sorted(splits):
        print(f"Found split '{name}': {splits[name]}")

    order = ["train"] + sorted(k for k in splits if k != "train")
    counts: dict[str, Counter] = {}
    for name in order:
        print(f"Copying '{name}' ...", flush=True)
        counts[name] = copy_split(splits[name], out_root / name)

    train_counts = counts["train"]
    print(f"\nClasses: {len(train_counts)}")
    header = f"{'class':<28}" + "".join(f"{n:>8}" for n in order)
    print(header)
    for class_name in sorted(train_counts):
        row = f"{class_name:<28}" + "".join(f"{counts[n].get(class_name, 0):>8}" for n in order)
        print(row)
    print(f"{'TOTAL':<28}" + "".join(f"{sum(counts[n].values()):>8}" for n in order))

    actual_total = sum(train_counts.values())
    if actual_total != 2400:
        print(f"\nNOTE: the assignment specifies 2400 training images, found {actual_total}.")

    if staging is not None and not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("\nRemoved staging folder data_raw/")
    print(f"\nDataset ready at {out_root.resolve()}")


if __name__ == "__main__":
    main()
