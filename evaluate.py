"""Score a saved classifier on the validation split or the test set.

    python evaluate.py --checkpoint runs/joint_seed0/best.pt --split val
    python evaluate.py --checkpoint runs/joint_seed0/best.pt --split test

Validation uses the same stratified split as training (the seed stored in the
checkpoint). The test set is only read when --split test is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from ssl_assignment.data import build_datasets, build_loaders  # noqa: E402
from ssl_assignment.engine import evaluate  # noqa: E402
from ssl_assignment.models import build_model, classifier_module  # noqa: E402
from ssl_assignment.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--data-root", default=None, help="Override the data root stored in the checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta", default=None, choices=[None, "none", "hflip"], help="Override checkpoint TTA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "classifier":
        raise SystemExit(
            "This file is not a classifier checkpoint. SimCLR pretraining saves last.pt; "
            "fine-tune it and then evaluate best.pt."
        )
    cfg = checkpoint["config"]
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.tta:
        cfg["eval"]["tta"] = args.tta

    if args.split == "test":
        print("Scoring the TEST set. Do this once, after validation has already selected the checkpoint.")

    bundle = build_datasets(cfg)
    if bundle["class_names"] != list(checkpoint["class_names"]):
        raise SystemExit(
            "Class order in the data folder does not match the checkpoint.\n"
            f"checkpoint: {checkpoint['class_names']}\n"
            f"data:       {bundle['class_names']}"
        )
    loaders = build_loaders(cfg, bundle)
    loader = loaders[args.split]
    if loader is None:
        raise SystemExit(f"No {args.split} split found under {cfg['data']['root']}.")

    model = build_model(cfg, num_classes=len(bundle["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    device = get_device(args.device)
    model.to(device)
    metrics = evaluate(model, loader, device, tta=str(cfg["eval"].get("tta", "none")))

    # Per-class counts from a second pass. 400 test images, so the extra pass is cheap
    # and it is the part worth reading: Assignment 1's errors were between similar scenes.
    scorer = classifier_module(model)
    scorer.eval()
    n_classes = len(bundle["class_names"])
    correct = [0] * n_classes
    total = [0] * n_classes
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            preds = scorer(images).argmax(dim=1).cpu()
            targets = targets.cpu()
            for pred, target in zip(preds.tolist(), targets.tolist()):
                total[target] += 1
                correct[target] += int(pred == target)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split} ({sum(total)} images)")
    print(f"Accuracy:   {metrics['acc'] * 100:.2f}%")
    print(f"Top-5:      {metrics['top5_acc'] * 100:.2f}%")
    print("Per class:")
    for index, name in enumerate(bundle["class_names"]):
        acc = correct[index] / total[index] if total[index] else 0.0
        print(f"  {name:<24} {correct[index]:>3}/{total[index]:<3}  {acc * 100:5.1f}%")


if __name__ == "__main__":
    main()
