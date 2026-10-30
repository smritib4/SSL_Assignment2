"""Train one experiment.

    python train.py --config configs/resnet18_supervised.yaml
    python train.py --config configs/simclr_pretrain.yaml
    python train.py --config configs/simclr_finetune_matched.yaml
    python train.py --config configs/joint.yaml

The test set is never opened here. Score it with evaluate.py after the
validation table has picked a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from ssl_assignment.config import load_config, save_config  # noqa: E402
from ssl_assignment.data import build_datasets, build_loaders, describe_dataset  # noqa: E402
from ssl_assignment.engine import fit  # noqa: E402
from ssl_assignment.models import (  # noqa: E402
    SimCLRModel,
    build_model,
    encoder_state_dict,
    freeze_encoder,
    load_encoder_weights,
)
from ssl_assignment.utils import (  # noqa: E402
    count_parameters,
    environment_info,
    get_device,
    run_directory,
    set_seed,
)

RESULTS_CSV = Path("experiments/results/results.csv")
LOG_ROOT = Path("experiments/run_logs")

CSV_FIELDS = [
    "experiment",
    "seed",
    "mode",
    "model",
    "img_size",
    "epochs",
    "epochs_completed",
    "interrupted",
    "freeze_encoder",
    "init_checkpoint",
    "params_trainable",
    "best_epoch",
    "val_acc",
    "val_top5",
    "ssl_loss",
    "train_time_s",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="key.subkey=value")
    parser.add_argument("--data-root", default=None, help="Override data.root")
    parser.add_argument("--device", default=None, help="cpu or cuda. Default: cuda if available")
    return parser.parse_args()


def tensors_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: tensors_to_cpu(value) for key, value in obj.items()}
    return obj


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    mode = str(cfg["experiment"]["mode"]).lower()
    if mode not in {"supervised", "pretrain", "joint"}:
        raise SystemExit(f"experiment.mode must be supervised, pretrain, or joint. Got {mode!r}.")

    set_seed(int(cfg["experiment"]["seed"]))
    device = get_device(args.device)
    bundle = build_datasets(cfg)
    loaders = build_loaders(cfg, bundle)
    model = build_model(cfg, num_classes=len(bundle["class_names"]))

    init_path = str(cfg["ssl"].get("init_checkpoint") or "").strip()
    if init_path:
        if mode != "supervised":
            raise SystemExit("ssl.init_checkpoint is only used with mode=supervised (finetune or linear probe).")
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        load_encoder_weights(model, checkpoint, expected_name=str(cfg["model"]["name"]))
        print(f"Loaded encoder from {init_path} (epoch {checkpoint.get('epoch')})", flush=True)

    if bool(cfg["model"].get("freeze_encoder", False)):
        if mode != "supervised":
            raise SystemExit("model.freeze_encoder is only meaningful for a supervised linear probe.")
        freeze_encoder(model)
        print("Encoder frozen. Training the linear head only.", flush=True)

    print(describe_dataset(bundle), flush=True)
    print(
        f"Mode: {mode} | device: {device} | trainable params: {count_parameters(model):,} "
        f"/ {count_parameters(model, trainable_only=False):,}",
        flush=True,
    )

    history_path = run_directory(cfg) / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    def on_epoch_end(record: dict) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    result = fit(model, loaders, cfg, device, on_epoch_end=on_epoch_end)
    trained = result["model"]
    run_dir = run_directory(cfg)

    if mode == "pretrain":
        if not isinstance(trained, SimCLRModel):
            raise RuntimeError("Pretrain mode did not produce a SimCLRModel.")
        payload = {
            "kind": "simclr",
            "encoder": encoder_state_dict(trained),
            "projector": trained.projector.state_dict(),
            "model_name": str(cfg["model"]["name"]),
            "feat_dim": trained.backbone.feat_dim,
            "class_names": bundle["class_names"],
            "config": cfg,
            "epoch": result["epochs_completed"],
            "ssl_loss": result["history"][-1]["train_loss"] if result["history"] else None,
            "environment": environment_info(),
        }
        ckpt_path = run_dir / "last.pt"
        torch.save(tensors_to_cpu(payload), ckpt_path)
        val_acc = ""
        val_top5 = ""
        ssl_loss = payload["ssl_loss"]
    else:
        trained.load_state_dict(result["final_state"])
        payload = {
            "kind": "classifier",
            "state_dict": trained.state_dict(),
            "model_name": str(cfg["model"]["name"]),
            "class_names": bundle["class_names"],
            "config": cfg,
            "epoch": result["best_epoch"],
            "val_acc": result["best_val_acc"],
            "val_top5": result["best_top5"],
            "environment": environment_info(),
        }
        ckpt_path = run_dir / "best.pt"
        torch.save(tensors_to_cpu(payload), ckpt_path)
        val_acc = f"{result['best_val_acc']:.4f}" if result["epochs_completed"] else ""
        val_top5 = f"{result['best_top5']:.4f}" if result["epochs_completed"] else ""
        ssl_loss = ""

    save_config(cfg, run_dir / "config.yaml")
    summary = {
        "experiment": cfg["experiment"]["name"],
        "mode": mode,
        "model": cfg["model"]["name"],
        "epochs_completed": result["epochs_completed"],
        "interrupted": result["interrupted"],
        "best_epoch": result["best_epoch"],
        "best_val_acc": result["best_val_acc"],
        "best_top5": result["best_top5"],
        "train_time_s": result["train_time_s"],
        "checkpoint": str(ckpt_path),
        "trainable_params": count_parameters(trained),
        "environment": environment_info(),
        "notes": cfg["experiment"].get("notes", ""),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_dir = LOG_ROOT / run_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, log_dir / "config.yaml")
    if history_path.exists():
        (log_dir / "history.jsonl").write_text(history_path.read_text(encoding="utf-8"), encoding="utf-8")
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    append_result(
        {
            "experiment": cfg["experiment"]["name"],
            "seed": cfg["experiment"]["seed"],
            "mode": mode,
            "model": cfg["model"]["name"],
            "img_size": cfg["data"]["img_size"],
            "epochs": cfg["optim"]["epochs"],
            "epochs_completed": result["epochs_completed"],
            "interrupted": result["interrupted"],
            "freeze_encoder": bool(cfg["model"].get("freeze_encoder", False)),
            "init_checkpoint": init_path,
            "params_trainable": count_parameters(trained),
            "best_epoch": result["best_epoch"],
            "val_acc": val_acc,
            "val_top5": val_top5,
            "ssl_loss": "" if ssl_loss is None else ssl_loss,
            "train_time_s": result["train_time_s"],
            "run_dir": str(run_dir),
        }
    )
    print(f"Saved {ckpt_path}", flush=True)
    if result["interrupted"]:
        print("Run flagged incomplete.", flush=True)


if __name__ == "__main__":
    main()
