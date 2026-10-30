"""NT-Xent, the contrastive loss used by SimCLR.

Two augmented views of the same image are a positive pair. Every other image
in the batch is a negative. The projection vectors are L2-normalized, so the
dot product is cosine similarity. Temperature sharpens or softens the
distribution: lower values make the model focus on the hardest negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return the mean NT-Xent loss for a batch of paired projections.

    `z1` and `z2` are `[N, D]` projections of the two views. `N` must be at
    least 2, because a batch of one image has no negatives.
    """
    if z1.ndim != 2 or z2.shape != z1.shape:
        raise ValueError(f"Expected paired projections of shape [N, D], got {tuple(z1.shape)} and {tuple(z2.shape)}")
    n = z1.size(0)
    if n < 2:
        raise ValueError(
            "NT-Xent needs at least 2 images in the batch. "
            "Increase batch_size and keep drop_last enabled for pretrain/joint."
        )

    # fp32: a large negative mask overflows fp16 before softmax.
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    z = torch.cat([z1, z2], dim=0)
    logits = z @ z.T / float(temperature)
    # A view is not its own negative. masked_fill (not the in-place version)
    # so autograd still sees the similarity matrix.
    self_mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1.0e4)
    # View 1's positive is the matching view 2, and the reverse.
    labels = torch.cat(
        [
            torch.arange(n, device=z.device) + n,
            torch.arange(n, device=z.device),
        ]
    )
    return F.cross_entropy(logits, labels)
