import json
from pathlib import Path

import numpy as np
import torch

from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2
from vidfm3d.models.components.probe_path_integration import PathIntegrationProbe
from vidfm3d.utils.spatial_diag import (
    compute_strict_hidden_object_target_for_id,
    robust_past_only_world_centroid,
)


def _strict_geometry():
    masks = torch.zeros(4, 4, 4, dtype=torch.long)
    masks[0, :2, :2] = 7
    masks[1, :2, :2] = 7
    points = torch.zeros(4, 4, 4, 3)
    points[..., 2] = 2.0
    points[0, :2, :2, 0] = 1.0
    points[1, :2, :2, 0] = 3.0
    confidence = torch.ones(4, 4, 4)
    extrinsics = torch.eye(4)[:3].repeat(4, 1, 1)
    return masks, points, confidence, extrinsics


def test_strict_b2_requires_exact_zero_target_pixels():
    masks, points, confidence, extrinsics = _strict_geometry()
    target = compute_strict_hidden_object_target_for_id(
        7, masks, points, confidence, extrinsics, min_visible_pixels=4
    )
    assert target is not None
    assert int(target["target_raw_pixels"].item()) == 0
    masks[-1, 0, 0] = 7
    assert compute_strict_hidden_object_target_for_id(
        7, masks, points, confidence, extrinsics, min_visible_pixels=4
    ) is None


def test_strict_centroid_never_reads_target_object_pixels():
    masks, points, confidence, _extrinsics = _strict_geometry()
    centroid, _ = robust_past_only_world_centroid(
        7, masks, points, confidence, min_visible_pixels=4
    )
    mutated_masks = masks.clone()
    mutated_points = points.clone()
    mutated_masks[-1] = 7
    mutated_points[-1, ..., 0] = 10000.0
    mutated, _ = robust_past_only_world_centroid(
        7, mutated_masks, mutated_points, confidence, min_visible_pixels=4
    )
    assert torch.equal(centroid, mutated)


def test_strict_centroid_changes_when_past_mask_changes():
    masks, points, confidence, _extrinsics = _strict_geometry()
    before, _ = robust_past_only_world_centroid(
        7, masks, points, confidence, min_visible_pixels=4
    )
    masks[0] = 0
    after, _ = robust_past_only_world_centroid(
        7, masks, points, confidence, min_visible_pixels=4
    )
    assert not torch.equal(before, after)








def test_conditioning_variants_keep_parameter_counts_matched():
    c1 = [ActionDynamicsProbe(8, hidden_dim=16, num_layers=1, num_heads=4, conditioning=mode)
          for mode in ("full", "visual_only", "action_only")]
    c2 = [PathIntegrationProbe(8, hidden_dim=16, num_layers=1, conditioning=mode)
          for mode in ("full", "visual_only", "action_only")]
    b2 = [EgoBeliefProbeV2(
        8, hidden_dim=16, num_layers=1, num_heads=4, max_seq_len=4,
        max_h=4, max_w=4, context_mode=mode,
    ) for mode in ("full", "query_only")]
    for probes in (c1, c2, b2):
        assert len({sum(parameter.numel() for parameter in probe.parameters()) for probe in probes}) == 1


def test_path_integration_public_state_update_matches_forward():
    torch.manual_seed(7)
    probe = PathIntegrationProbe(
        in_channels=8, action_dim=9, hidden_dim=16, num_layers=2, dropout=0.0
    ).eval()
    context = torch.randn(3, 4, 2, 2, 8)
    actions = torch.randn(3, 2, 9)
    state = probe.initialize_state(context)
    first = probe.update_state(state, actions[:, 0])
    second = probe.update_state(first, actions[:, 1])
    manual = torch.stack((probe.decode_state(first), probe.decode_state(second)), dim=1)
    assert torch.equal(manual, probe(context, actions))


def test_path_integration_direct_and_sequential_use_same_closed_state_type():
    probe = PathIntegrationProbe(
        in_channels=8, action_dim=9, hidden_dim=16, num_layers=2, dropout=0.0
    ).eval()
    context = torch.randn(2, 3, 1, 1, 8)
    actions = torch.randn(2, 2, 9)
    state = probe.initialize_state(context)
    sequential, sequential_state = probe.rollout_from_state(state, actions)
    direct_state = probe.update_state(state, torch.randn(2, 9))
    direct = probe.decode_state(direct_state)
    assert sequential.shape == (2, 2, 8)
    assert direct.shape == (2, 8)
    assert len(sequential_state) == len(direct_state) == 2
    assert all(left.shape == right.shape for left, right in zip(sequential_state, direct_state))
