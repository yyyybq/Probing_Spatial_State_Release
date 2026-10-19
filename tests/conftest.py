"""Shared pytest configuration.

The suite runs on CPU with synthetic inputs only; no dataset, cached features,
checkpoints or model weights are required.
"""

import os

# Keep Hydra/Lightning quiet and deterministic under pytest.
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
