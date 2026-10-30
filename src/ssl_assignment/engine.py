"""Optimisation and the three training loops: supervised, SimCLR, joint."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import nt_xent
from .models import JointModel, SimCLRModel, classifier_module
from .utils import AverageMeter


def _is_head_param(name: str) -> bool:
    return name.startswith("head.") or ".head." in name


def build_optimizer(model: nn.Module, optim_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW by default. Biases and norm scales are not decayed.

    `head_lr_mult` scales every parameter of the linear classifier. Encoder
    and projection-head parameters stay at the base learning rate. That split
    matters once an encoder already holds SimCLR features; a from-scratch run
    leaves the multiplier at 1.
    """
    base_lr = float(optim_cfg["lr"])
    head_lr = base_lr * float(optim_cfg.get("head_lr_mult", 1.0))
    weight_decay = float(optim_cfg["weight_decay"])
    groups: dict[str, dict[str, Any]] = {
        "backbone_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": head_lr, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = f"{'head' if _is_head_param(name) else 'backbone'}_{'nodecay' if no_decay else 'decay'}"
        groups[key]["params"].append(param)
    param_groups = [group for group in groups.values() if group["params"]]
    if not param_groups:
        raise ValueError("No trainable parameters. If freeze_encoder is true, the head should still train.")

    name = str(optim_cfg.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr)
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            nesterov=bool(optim_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unknown optimizer '{name}' (use adamw, adam, or sgd)")


def build_scheduler(
    optimizer: torch.optim.Optimizer, optim_cfg: dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    kind = str(optim_cfg.get("scheduler", "cosine")).lower()
    if kind in {"none", "constant", ""}:
        return None
    epochs = int(optim_cfg["epochs"])
    warmup_steps = int(float(optim_cfg.get("warmup_epochs", 0)) * steps_per_epoch)
    total_steps = max(1, epochs * steps_per_epoch)
    min_lr_ratio = float(optim_cfg.get("min_lr", 0.0)) / max(float(optim_cfg["lr"]), 1e-12)
    min_lr_ratio = min(min_lr_ratio, 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unknown scheduler '{kind}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _frozen_encoder_eval(model: nn.Module, freeze_encoder: bool) -> None:
    model.train()
    if not freeze_encoder:
        return
    # Batch-norm must keep the statistics learned in pretraining.
    if isinstance(model, JointModel):
        model.backbone.encoder.eval()
    elif hasattr(model, "encoder"):
        model.encoder.eval()


def _step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_clip: float,
) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss ({loss.item()}). Lower the learning rate and rerun.")
    scaler.scale(loss).backward()
    if grad_clip and grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().sum().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    tta: str = "none",
) -> dict[str, float]:
    """Score a classifier. Joint models are scored through the backbone only."""
    scorer = classifier_module(model)
    scorer.eval()
    loss_meter = AverageMeter()
    correct = 0
    correct_top5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = scorer(images)
        if tta == "hflip":
            logits = 0.5 * (logits + scorer(torch.flip(images, dims=[3])))
        if criterion is not None:
            loss_meter.update(float(criterion(logits, targets).item()), images.size(0))
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        k = min(5, logits.size(1))
        topk = logits.topk(k, dim=1).indices
        correct_top5 += int((topk == targets.unsqueeze(1)).any(dim=1).sum().item())
        total += images.size(0)
    return {
        "loss": loss_meter.avg if criterion is not None else 0.0,
        "acc": correct / total if total else 0.0,
        "top5_acc": correct_top5 / total if total else 0.0,
    }


def _train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_clip: float,
    freeze_encoder: bool,
    amp: bool,
) -> dict[str, float]:
    _frozen_encoder_eval(model, freeze_encoder)
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        _step(loss, model, optimizer, scaler, scheduler, grad_clip)
        loss_meter.update(float(loss.detach().item()), images.size(0))
        correct += _accuracy(logits.detach(), targets)
        total += images.size(0)
    return {
        "train_loss": loss_meter.avg,
        "train_acc": correct / total if total else 0.0,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def _paired_forward(model: nn.Module, view1: torch.Tensor, view2: torch.Tensor):
    """One forward on the concatenated views so batch-norm sees both."""
    packed = torch.cat([view1, view2], dim=0)
    return model(packed)


def _train_pretrain_epoch(
    model: SimCLRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_clip: float,
    temperature: float,
    amp: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    for view1, view2, _labels in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            projections = _paired_forward(model, view1, view2)
            z1, z2 = projections.chunk(2, dim=0)
            loss = nt_xent(z1, z2, temperature)
        _step(loss, model, optimizer, scaler, scheduler, grad_clip)
        loss_meter.update(float(loss.detach().item()), view1.size(0))
    return {
        "train_loss": loss_meter.avg,
        "train_acc": 0.0,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def _train_joint_epoch(
    model: JointModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_clip: float,
    temperature: float,
    lambda_contrast: float,
    amp: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    ssl_meter = AverageMeter()
    correct = 0
    total = 0
    for view1, view2, labels in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits, projections = _paired_forward(model, view1, view2)
            logits1, logits2 = logits.chunk(2, dim=0)
            z1, z2 = projections.chunk(2, dim=0)
            ce = 0.5 * (criterion(logits1, labels) + criterion(logits2, labels))
            contrast = nt_xent(z1, z2, temperature)
            loss = ce + float(lambda_contrast) * contrast
        _step(loss, model, optimizer, scaler, scheduler, grad_clip)
        batch = view1.size(0)
        loss_meter.update(float(loss.detach().item()), batch)
        ce_meter.update(float(ce.detach().item()), batch)
        ssl_meter.update(float(contrast.detach().item()), batch)
        correct += _accuracy(logits1.detach(), labels)
        total += batch
    return {
        "train_loss": loss_meter.avg,
        "train_ce": ce_meter.avg,
        "train_ssl": ssl_meter.avg,
        "train_acc": correct / total if total else 0.0,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader | None],
    cfg: dict[str, Any],
    device: torch.device,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train, keeping the best validation accuracy for any run that classifies.

    SimCLR pretraining has no classifier yet, so it keeps the final encoder.
    KeyboardInterrupt returns the best weights so far and flags the run
    incomplete. A partial run must not be compared with a finished one.
    """
    optim_cfg = cfg["optim"]
    ssl_cfg = cfg["ssl"]
    mode = str(cfg["experiment"]["mode"]).lower()
    epochs = int(optim_cfg["epochs"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    if train_loader is None or val_loader is None:
        raise RuntimeError("Train and validation loaders are required.")
    if len(train_loader) == 0:
        raise RuntimeError(
            "Training loader is empty. The batch size is larger than the training split "
            "after drop_last. Lower data.batch_size."
        )

    model = model.to(device)
    freeze = bool(cfg["model"].get("freeze_encoder", False)) and mode == "supervised"
    criterion = nn.CrossEntropyLoss(label_smoothing=float(optim_cfg.get("label_smoothing", 0.0) or 0.0))
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, len(train_loader))
    amp = bool(optim_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    grad_clip = float(optim_cfg.get("grad_clip", 0.0) or 0.0)
    temperature = float(ssl_cfg["temperature"])
    lambda_contrast = float(ssl_cfg["lambda_contrast"])
    tta = str(cfg["eval"].get("tta", "none"))

    best_acc = -1.0
    best_top5 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, Any]] = []
    start = time.time()
    interrupted = False

    try:
        for epoch in range(1, epochs + 1):
            if mode == "pretrain":
                train_stats = _train_pretrain_epoch(
                    model, train_loader, optimizer, device, scaler, scheduler, grad_clip, temperature, amp
                )
                record = {"epoch": epoch, **train_stats, "elapsed_s": round(time.time() - start, 1)}
                print(
                    f"epoch {epoch:03d}/{epochs} | lr {record['lr']:.2e} | "
                    f"nt-xent {record['train_loss']:.4f} | {record['elapsed_s']:.0f}s",
                    flush=True,
                )
            elif mode == "joint":
                train_stats = _train_joint_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    scheduler,
                    grad_clip,
                    temperature,
                    lambda_contrast,
                    amp,
                )
                val_stats = evaluate(model, val_loader, device, criterion, tta=tta)
                record = {
                    "epoch": epoch,
                    **train_stats,
                    "val_loss": val_stats["loss"],
                    "val_acc": val_stats["acc"],
                    "val_top5": val_stats["top5_acc"],
                    "elapsed_s": round(time.time() - start, 1),
                }
                if val_stats["acc"] > best_acc:
                    best_acc = val_stats["acc"]
                    best_top5 = val_stats["top5_acc"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                record["best_val_acc"] = best_acc
                print(
                    f"epoch {epoch:03d}/{epochs} | lr {record['lr']:.2e} | "
                    f"loss {record['train_loss']:.4f} (ce {record['train_ce']:.4f} + "
                    f"{lambda_contrast:g} * nt-xent {record['train_ssl']:.4f}) | "
                    f"val acc {record['val_acc']:.4f} | best {best_acc:.4f} | {record['elapsed_s']:.0f}s",
                    flush=True,
                )
            elif mode == "supervised":
                train_stats = _train_supervised_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    scheduler,
                    grad_clip,
                    freeze,
                    amp,
                )
                val_stats = evaluate(model, val_loader, device, criterion, tta=tta)
                record = {
                    "epoch": epoch,
                    **train_stats,
                    "val_loss": val_stats["loss"],
                    "val_acc": val_stats["acc"],
                    "val_top5": val_stats["top5_acc"],
                    "elapsed_s": round(time.time() - start, 1),
                }
                if val_stats["acc"] > best_acc:
                    best_acc = val_stats["acc"]
                    best_top5 = val_stats["top5_acc"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                record["best_val_acc"] = best_acc
                print(
                    f"epoch {epoch:03d}/{epochs} | lr {record['lr']:.2e} | "
                    f"train loss {record['train_loss']:.4f} acc {record['train_acc']:.4f} | "
                    f"val acc {record['val_acc']:.4f} | best {best_acc:.4f} | {record['elapsed_s']:.0f}s",
                    flush=True,
                )
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            history.append(record)
            if on_epoch_end is not None:
                on_epoch_end(record)
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\nInterrupted after {len(history)} of {epochs} epochs. "
            "The checkpoint will be the best validation point so far, and the run is flagged incomplete.",
            flush=True,
        )

    if mode == "pretrain":
        # Last completed epoch. There is no validation accuracy to select on.
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_acc = -1.0
    else:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_val_acc": best_acc,
        "best_top5": best_top5,
        "best_epoch": best_epoch,
        "train_time_s": round(time.time() - start, 1),
        "epochs_completed": len(history),
        "interrupted": interrupted,
        "final_state": best_state,
    }
