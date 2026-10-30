"""Train and eval transforms.

Classification training and SimCLR training do not share an augmentation
policy. Assignment 1's strong preset (RandAugment, heavy erasing) was aimed at
an ImageNet-initialized network and did not help there. From-scratch training
needs milder classification augmentation. SimCLR needs a different policy
again: two views of one image must disagree on colour and crop, or the
contrastive task is trivial, but the crops cannot be so small that a scene
category disappears. Random erasing is left off for SimCLR because it deletes
the region the two views are supposed to agree on.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def _blur_kernel(img_size: int) -> int:
    kernel = max(3, int(round(img_size * 0.1)))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def build_eval_transform(data_cfg: dict[str, Any]) -> transforms.Compose:
    img_size = int(data_cfg["img_size"])
    resize = int(round(img_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(data_cfg["mean"]), std=list(data_cfg["std"])),
        ]
    )


def build_train_transform(data_cfg: dict[str, Any], aug_cfg: dict[str, Any]) -> transforms.Compose:
    img_size = int(data_cfg["img_size"])
    scale = aug_cfg.get("train_crop_scale", [0.65, 1.0])
    ops: list[Any] = [
        transforms.RandomResizedCrop(img_size, scale=(float(scale[0]), float(scale[1])), antialias=True),
    ]
    hflip = float(aug_cfg.get("hflip", 0.5) or 0.0)
    if hflip > 0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip))

    jitter = float(aug_cfg.get("color_jitter", 0.0) or 0.0)
    jitter_p = float(aug_cfg.get("jitter_p", 1.0) or 0.0)
    if jitter > 0 and jitter_p > 0:
        hue = min(0.5, float(aug_cfg.get("jitter_hue", 0.05) or 0.0))
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=jitter,
                        contrast=jitter,
                        saturation=jitter,
                        hue=hue,
                    )
                ],
                p=jitter_p,
            )
        )

    grayscale_p = float(aug_cfg.get("grayscale_p", 0.0) or 0.0)
    if grayscale_p > 0:
        ops.append(transforms.RandomGrayscale(p=grayscale_p))

    blur_p = float(aug_cfg.get("blur_p", 0.0) or 0.0)
    if blur_p > 0:
        sigma = aug_cfg.get("blur_sigma", [0.1, 2.0])
        ops.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=_blur_kernel(img_size),
                        sigma=(float(sigma[0]), float(sigma[1])),
                    )
                ],
                p=blur_p,
            )
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=list(data_cfg["mean"]), std=list(data_cfg["std"])))

    erasing = float(aug_cfg.get("random_erasing", 0.0) or 0.0)
    if erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing, value="random"))
    return transforms.Compose(ops)
