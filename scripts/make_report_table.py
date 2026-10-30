"""Print the experiment table from experiments/results/results.csv.

The last completed row for each experiment name is the one that counts.
Interrupted runs stay visible and are not eligible for selection.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("experiments/results/results.csv")

# Display order is the experiment ladder, not the order the runs happened to finish.
ORDER = [
    "smallcnn_supervised",
    "resnet18_supervised",
    "simclr_pretrain",
    "simclr_linear_probe",
    "simclr_finetune_matched",
    "simclr_finetune",
    "joint",
    "best",
]


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"No results file at {RESULTS}. Train something first.")
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("results.csv has a header and no rows.")

    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        latest[row["experiment"]] = row

    names = [name for name in ORDER if name in latest]
    names.extend(name for name in latest if name not in names)

    print("| Experiment | Mode | Model | Epochs | Best epoch | Val acc | Notes |")
    print("|---|---|---|---:|---:|---:|---|")
    candidates: list[tuple[float, str, str]] = []
    for name in names:
        row = latest[name]
        interrupted = _as_bool(row.get("interrupted", ""))
        completed = row.get("epochs_completed", "")
        planned = row.get("epochs", "")
        partial = interrupted or (completed != planned)
        acc_text = row.get("val_acc", "") or ""
        if row.get("mode") == "pretrain":
            shown = "n/a"
            note = f"final NT-Xent {row.get('ssl_loss', '')}"
        elif acc_text:
            shown = f"{float(acc_text) * 100:.2f}%"
            note = "incomplete" if partial else ""
        else:
            shown = ""
            note = "incomplete" if partial else ""
        if acc_text and not partial and row.get("mode") != "pretrain":
            candidates.append((float(acc_text), name, row.get("run_dir", "")))
        print(
            f"| {name} | {row.get('mode', '')} | {row.get('model', '')} | "
            f"{completed}/{planned} | {row.get('best_epoch', '')} | {shown} | {note} |"
        )

    print()
    if not candidates:
        print("No completed classification run to select yet.")
        return
    candidates.sort(reverse=True)
    best_acc, best_name, best_dir = candidates[0]
    print(
        f"Selection rule: highest validation accuracy among completed classification runs.\n"
        f"Current leader: {best_name} at {best_acc * 100:.2f}%  ({best_dir})"
    )
    print("Score that checkpoint on the test set once:")
    print(f"  python evaluate.py --checkpoint {best_dir}/best.pt --split test")
    if len(candidates) > 1 and (best_acc - candidates[1][0]) < 0.01:
        other = candidates[1]
        print(
            f"The next run ({other[1]}, {other[0] * 100:.2f}%) is within 1 point. "
            "On 480 validation images that gap is inside the noise floor measured in "
            "Assignment 1 (about 0.6 points). Score both and report both."
        )


if __name__ == "__main__":
    main()
