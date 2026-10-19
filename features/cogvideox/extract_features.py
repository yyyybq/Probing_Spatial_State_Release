"""CogVideoX helpers used by features/run_inscene15k.py.

``run_inscene15k.py`` imports :func:`forward_cogvideox`. It splits a 97-frame
window into two 49-frame clips (frame 0 plus the odd frames 1..95, and frame 0
plus the even frames 2..96), runs the featurizer on each clip and concatenates
the per-layer features along dim 0.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image

from .cogvideox_feature import CogVideoXFeaturizer


def forward_cogvideox(
    model: CogVideoXFeaturizer,
    frames: List[Image.Image],
    t: int,
    layer_ids: List[int],
    noise_generator: torch.Generator | None = None,
) -> Dict[int, torch.Tensor]:
    # 1.  Build 2 clips, each 49 frames:
    #     clip_0 = [0, 1, 3, 5, ..., 95]
    #     clip_1 = [0, 2, 4, 6, ..., 96]
    clip_indices = [
        [0] + [i + 2 * k for k in range(48)] for i in range(1, 3)  # 1≤i≤2
    ]  # list[2][49]

    # get the frames for each clip
    clips = [[frames[i] for i in indices] for indices in clip_indices]

    all_features = []
    for i, clip in enumerate(clips):
        if len(clip) != 49:
            raise RuntimeError(f"Clip {i} has {len(clip)} frames, expected 49.")

        with torch.no_grad():
            feat = model.forward(
                video=clip,
                t=t,
                output_layer_indices=layer_ids,
                noise_generator=noise_generator,
            )
            all_features.append(feat)

    # 2. Merge the feature dictionaries from both clips:
    merged_features = {
        layer_id: torch.cat(
            [all_features[0][layer_id], all_features[1][layer_id]], dim=0
        )
        for layer_id in all_features[0].keys()
    }
    return merged_features
