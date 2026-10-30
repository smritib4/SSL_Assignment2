"""Write predictions for a folder of unlabelled images.

    python predict.py --checkpoint runs/joint_seed0/best.pt --images-dir data/test2 --out reports/test2.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ssl_assignment.data import FlatImageDataset  # noqa: E402
from ssl_assignment.models import build_model, classifier_module  # noqa: E402
from ssl_assignment.transforms import build_eval_transform  # noqa: E402
from ssl_assignment.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "classifier":
        raise SystemExit("predict.py needs a classifier checkpoint (best.pt), not a SimCLR encoder.")
    cfg = checkpoint["config"]
    dataset = FlatImageDataset(args.images_dir, transform=build_eval_transform(cfg["data"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(cfg, num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    device = get_device(args.device)
    scorer = classifier_module(model).to(device).eval()
    class_names = list(checkpoint["class_names"])

    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for images, names in loader:
            preds = scorer(images.to(device)).argmax(dim=1).cpu().tolist()
            rows.extend((name, class_names[index]) for name, index in zip(names, preds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
