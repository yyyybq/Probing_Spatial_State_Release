"""Fast3R helpers used by features/run_inscene15k.py.

``run_inscene15k.py`` imports :func:`load_model`, :func:`vidfm3d_forward` and
the token-grid constants ``HTOK``, ``WTOK`` and ``C``. ``vidfm3d_forward``
loads the frames with the vendored ``f3r.dust3r.utils.image.load_images``
(fixed 512x288 resize) and returns the per-layer decoder features, each
reshapeable to ``(T, HTOK, WTOK, C)`` = ``(T, 18, 32, 1024)``.

The vendored Fast3R package lives in ``features/f3r/f3r``; see
``features/f3r/LICENSE`` and ``features/f3r/MODIFICATIONS.md``.
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Sequence

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f3r.dust3r.inference_multiview import inference
from f3r.dust3r.utils.image import load_images
from f3r.models.fast3r import Fast3R


# --------------------------------------------------------------------------- #
# Token grid of the fixed 512x288 input (patch size 16)                        #
# --------------------------------------------------------------------------- #

LONG_SIDE = 512
HTOK = 288 // 16  # 18
WTOK = 512 // 16  # 32
C = 1024  # decoder channel dim


@lru_cache(maxsize=None)
def load_model(model_id: str, device: str) -> Fast3R:
    """Load Fast3R once (cached across calls)."""
    logging.debug("Loading model %s on %s ...", model_id, device)
    model = Fast3R.from_pretrained(model_id).to(device).eval()
    return model


@torch.no_grad()
def vidfm3d_forward(
    filelist: Sequence[str],
    model: Fast3R,
    device: torch.device,
    random_generator: torch.Generator | None = None,
):
    """load_images -> inference (bfloat16) -> per-layer decoder features."""
    imgs = load_images(filelist, size=LONG_SIDE, verbose=False)
    ret = inference(
        imgs,
        model,
        device,
        dtype=torch.bfloat16,  # stay hardcoded
        verbose=False,
        profiling=False,
        random_generator=random_generator,
    )
    # Handle accidentally returning (features, profiling)
    if isinstance(ret, tuple):
        ret = ret[0]
    return ret  # expected: list[Tensor] where each is (T*HTOK*WTOK*C) compatible
