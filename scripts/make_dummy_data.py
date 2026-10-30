"""Synthetic 16-class images so the training loop can be checked without the dataset.

    python scripts/make_dummy_data.py --out data_dummy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data_dummy")
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    out = Path(args.out)
    for split, count in (("train", args.train_per_class), ("test", args.test_per_class)):
        for class_id in range(16):
            folder = out / split / f"class_{class_id:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = rng.integers(0, 40, size=(96, 96, 3), dtype=np.uint8)
                channel = class_id % 3
                boosted = image[:, :, channel].astype(np.int16) + 120 + 6 * class_id
                image[:, :, channel] = np.clip(boosted, 0, 255).astype(np.uint8)
                y0 = (class_id * 5) % 70
                x0 = (class_id * 7) % 70
                image[y0 : y0 + 18, x0 : x0 + 18] = (30 * class_id) % 255
                Image.fromarray(image, mode="RGB").save(folder / f"img_{index}.jpg")
    print(f"Wrote dummy data to {out.resolve()}")


if __name__ == "__main__":
    main()
