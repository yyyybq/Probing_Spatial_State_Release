"""Every public entry point composes its Hydra config from any working directory.

``--cfg job`` composes the config through the real decorated entry point without
loading data or checkpoints, so it also catches a misplaced ``@hydra.main``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("vidfm3d/train.py", "inscene15k_streaming/ego_belief_v2_dino_v1", []),
    ("vidfm3d/eval_diag.py", "inscene15k_streaming/view_consistency_dino_v1", []),
    ("vidfm3d/eval_diag.py", "inscene15k_streaming/action_dynamics_dino_h124_v1",
     ["++evaluation_protocol=c3_action_sensitivity"]),
    ("vidfm3d/eval_a1_streaming.py", "inscene15k_streaming/streaming_depth_dino_v1", []),
]


@pytest.mark.parametrize("entry,experiment,extra", CASES)
def test_entrypoint_composes_without_project_root(tmp_path, entry, experiment, extra):
    env = {k: v for k, v in os.environ.items() if k != "PROJECT_ROOT"}
    env.update(INSCENE_DATA_ROOT=str(tmp_path), INSCENE_STREAMING_FEAT_ROOT=str(tmp_path),
               WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, str(ROOT / entry), f"experiment={experiment}",
           "+eval_split=val", "+ckpt_path='/x/epoch=0-step=1.ckpt'", *extra, "--cfg", "job"]
    out = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"job_name:" in out.stdout
