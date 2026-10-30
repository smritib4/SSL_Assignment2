"""From-scratch backbones and the SimCLR projection head.

No constructor in this module accepts pretrained weights. ResNet-18 is built
with `weights=None`. The SmallCNN matches the 8-convolution network from
Assignment 1, which is the from-scratch result worth comparing against
(89.38% validation at 128px). ImageNet ConvNeXt / ResNet fine-tunes from that
assignment are not usable here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class Backbone(nn.Module):
    """Encoder plus a linear classifier on the pre-projection representation."""

    encoder: nn.Module
    drop: nn.Module
    head: nn.Linear
    feat_dim: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x)))


class SmallCNN(Backbone):
    """Four stages of conv-BN-ReLU twice, then 2x pool, then global average pool.

    This is the Assignment 1 from-scratch architecture, split so `encode()`
    returns the 512-d pooled vector and `head` is only the linear layer.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        widths = (64, 128, 256, 512)
        blocks: list[nn.Module] = []
        prev = 3
        for width in widths:
            blocks.extend(
                [
                    nn.Conv2d(prev, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
            )
            prev = width
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.feat_dim = widths[-1]
        self.head = nn.Linear(self.feat_dim, num_classes)


class ResNet18Scratch(Backbone):
    """torchvision ResNet-18 with a randomly initialized stem and a new head."""

    def __init__(self, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        net = models.resnet18(weights=None)
        self.feat_dim = int(net.fc.in_features)
        net.fc = nn.Identity()
        self.encoder = net
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, num_classes)


class ProjectionHead(nn.Module):
    """SimCLR MLP. Discarded after pretraining; the loss is applied to its output.

    The representation used for classification is the encoder output, not this
    128-d vector. The paper's ablation shows the layer before the projection
    transfers better.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, backbone: Backbone, proj_hidden: int, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(backbone.feat_dim, proj_hidden, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone.encode(x))


class JointModel(nn.Module):
    """Classification on the encoder output, NT-Xent on the projection.

    Dropout sits on the classifier path only. The projection should see the
    clean representation, otherwise the contrastive target itself is noise.
    """

    def __init__(self, backbone: Backbone, proj_hidden: int, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(backbone.feat_dim, proj_hidden, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone.encode(x)
        logits = self.backbone.head(self.backbone.drop(h))
        z = self.projector(h)
        return logits, z


def build_backbone(name: str, num_classes: int, dropout: float) -> Backbone:
    key = name.lower().strip()
    if key in {"smallcnn", "small_cnn"}:
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if key in {"resnet18", "resnet_18"}:
        return ResNet18Scratch(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        f"Unknown model '{name}'. This assignment only allows from-scratch "
        "models defined here: resnet18, smallcnn."
    )


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    """Build the module that `fit` trains for this experiment mode."""
    model_cfg = cfg["model"]
    ssl_cfg = cfg["ssl"]
    backbone = build_backbone(
        str(model_cfg["name"]),
        num_classes=num_classes,
        dropout=float(model_cfg.get("dropout", 0.0) or 0.0),
    )
    mode = str(cfg["experiment"]["mode"]).lower()
    if mode == "pretrain":
        # The classification head is not part of SimCLR. Leave it frozen so
        # its random weights are not saved as if they had been trained.
        for param in backbone.head.parameters():
            param.requires_grad_(False)
        return SimCLRModel(
            backbone,
            proj_hidden=int(ssl_cfg["proj_hidden"]),
            proj_dim=int(ssl_cfg["proj_dim"]),
        )
    if mode == "joint":
        return JointModel(
            backbone,
            proj_hidden=int(ssl_cfg["proj_hidden"]),
            proj_dim=int(ssl_cfg["proj_dim"]),
        )
    if mode == "supervised":
        return backbone
    raise ValueError(f"Unknown experiment.mode '{mode}'. Use supervised, pretrain, or joint.")


def freeze_encoder(model: nn.Module) -> None:
    """Stop encoder gradients. The linear head (and only the head) keeps training."""
    encoder = _encoder_module(model)
    for param in encoder.parameters():
        param.requires_grad_(False)


def _encoder_module(model: nn.Module) -> nn.Module:
    if isinstance(model, (SimCLRModel, JointModel)):
        return model.backbone.encoder
    if isinstance(model, Backbone):
        return model.encoder
    raise TypeError(f"Cannot find an encoder on {type(model).__name__}")


def load_encoder_weights(model: nn.Module, checkpoint: dict[str, Any], expected_name: str) -> None:
    """Copy a SimCLR encoder into a freshly built backbone. The head stays random."""
    if checkpoint.get("kind") != "simclr":
        raise ValueError(
            "init_checkpoint must be a SimCLR pretrain checkpoint (kind='simclr'), "
            f"got kind={checkpoint.get('kind')!r}."
        )
    ckpt_name = str(checkpoint.get("model_name", ""))
    if ckpt_name != expected_name:
        raise ValueError(
            f"Checkpoint backbone is '{ckpt_name}' but this config asks for '{expected_name}'."
        )
    encoder = _encoder_module(model)
    encoder.load_state_dict(checkpoint["encoder"])


def encoder_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return _encoder_module(model).state_dict()


def classifier_module(model: nn.Module) -> nn.Module:
    """The module whose `forward` returns class logits. Used at evaluation."""
    if isinstance(model, JointModel):
        return model.backbone
    if isinstance(model, Backbone):
        return model
    raise TypeError(f"{type(model).__name__} has no classifier. Pretrain checkpoints are not classifiers.")
