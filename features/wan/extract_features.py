"""Wan helpers used by features/run_inscene15k.py.

``run_inscene15k.py`` imports :func:`reshape_to_t_h_w_c`, which turns the Wan
transformer output ``(1, N_tokens, C)`` of an 81-frame 480x832 window into
``(21, 30, 52, C)``. :func:`forward_wan` is a thin wrapper around
``WanFeaturizer.forward``.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image

from .wan_feature import WanFeaturizer
from .wan_feature_i2v import WanFeaturizerI2V


def forward_wan(
    model: WanFeaturizer | WanFeaturizerI2V,
    frames: List[Image.Image],
    prompt: str,
    t: int,
    layer_ids: List[int],
    ensemble: int,
) -> Dict[int, torch.Tensor]:
    with torch.no_grad():
        return model.forward(
            video=frames,
            prompt=prompt,
            t=t,
            output_layer_indices=layer_ids,
            ensemble_size=ensemble,
        )


def reshape_to_t_h_w_c(raw: torch.Tensor) -> torch.Tensor:
    """
    Wan (1, N_tokens, C) → (T, H, W, C)

    With 832x480 inputs Wan produces (H_p=30, W_p=52) spatial tokens per
    *output* frame and temporal stride = 4 ⇒ T = 80/4+1 = 21 frames.

    """
    t_tokens, h_tokens, w_tokens = 21, 30, 52
    assert raw.ndim == 3, f"Expected 3D tensor, got {raw.shape}"
    assert raw.shape[0] == 1, f"Expected batch size 1, got {raw.shape[0]}"
    assert (
        raw.shape[1] == t_tokens * h_tokens * w_tokens
    ), f"Expected {t_tokens * h_tokens * w_tokens} tokens, got {raw.shape[1]}"
    return (
        raw.squeeze(0)  # (N_tokens, C)
        .reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1])
        .contiguous()
    )
