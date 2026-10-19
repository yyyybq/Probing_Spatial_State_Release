"""DINOv2 helpers used by features/run_inscene15k.py.

``run_inscene15k.py`` resizes frames to 728x420 (W x H) itself, then imports
:func:`dino_forward` to run ``Dinov2Model`` on a batch of PIL images and
:func:`reshape_tokens` to drop the CLS token and reshape one frame's patch
tokens to ``(30, 52, C)``. :func:`load_model` is a cached model loader.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from transformers import Dinov2Model


@torch.no_grad()
def dino_forward(model, proc, pil_batch, device):
    """Return (B, 1+N, C) where N = 30*52."""
    inputs = proc(
        images=pil_batch,
        return_tensors="pt",
        do_center_crop=False,
        do_resize=False,  # already resized
    ).to(device)
    return model(**inputs).last_hidden_state  # (B,1+N,C)


def reshape_tokens(tok: torch.Tensor, hp=30, wp=52) -> torch.Tensor:
    """(1+N,C) -> (hp,wp,C) dropping CLS."""
    return tok[1:].reshape(hp, wp, -1).contiguous()


@lru_cache(maxsize=None)
def load_model(model_id: str, device: str) -> Dinov2Model:
    model = Dinov2Model.from_pretrained(model_id).to(device).eval()
    return model
