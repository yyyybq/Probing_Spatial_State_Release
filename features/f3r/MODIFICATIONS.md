# Modifications to the vendored Fast3R code

`features/f3r/f3r/` is derived from [Fast3R](https://github.com/facebookresearch/fast3r)
(Copyright (c) Meta Platforms, Inc. and affiliates), licensed under the FAIR
Noncommercial Research License in [`LICENSE`](LICENSE). It contains code from
[DUSt3R](https://github.com/naver/dust3r) and [CroCo](https://github.com/naver/croco)
(Copyright (c) Naver Corporation), licensed under CC BY-NC-SA 4.0; see
`f3r/dust3r/LICENSE`, `f3r/dust3r/NOTICE`, `f3r/croco/LICENSE` and `f3r/croco/NOTICE`.

This file lists the changes made to that code. It is the notice of modification
that CC BY-NC-SA 4.0, Section 3(a)(1)(B), requires for the adapted DUSt3R and
CroCo files.

## Provenance

- Upstream: https://github.com/facebookresearch/fast3r, package `fast3r/`.
- Route: Fast3R, then the copy in [VidFM3D](https://github.com/zxhuang1698/VidFM3D)
  at `features/f3r/f3r/`, then this repository.
- Upstream commit: not identified. The changes below were checked against the
  `main` branches of Fast3R and VidFM3D.

## Changes inherited from the VidFM3D copy

- The package is renamed from `fast3r` to `f3r`, and the imports inside the
  package are rewritten to match.
- `dust3r/inference_multiview.py`: `loss_of_one_batch` returns the model
  predictions only (no loss and no result dict), and `inference` returns that
  value directly.
- `models/fast3r.py`: `Fast3R.forward` gains a `return_feature` argument
  (default `True`). When it is set, `forward` returns the per-layer decoder
  outputs instead of the head predictions.
- `dust3r/utils/image.py`: for any `size` other than 224, `load_images`
  resizes every image to a fixed 512x288 with `PIL.Image.LANCZOS`. Upstream
  resizes the long side to `size`. The center crop that upstream applies after
  the resize is removed.
- The code is reformatted (black/isort style).

## Changes made in this repository

- Optional `random_generator` (`torch.Generator`) argument, threaded through
  `dust3r/inference_multiview.py` (`inference`, `loss_of_one_batch`) and
  `models/fast3r.py` (`Fast3R.forward`, the decoder `forward` methods,
  `_get_random_image_pos` and `_get_random_freqs_cis`). When a generator is
  passed, the random image-position ids are drawn from it, so feature
  extraction is deterministic. With the default `None` the upstream behaviour
  is unchanged.
- The tree is pruned to the 50 Python modules in the import closure of
  `features/f3r/extract_features.py`, plus `croco/LICENSE` and `croco/NOTICE`
  and the two empty package files `data/__init__.py` and
  `data/components/__init__.py`. Training, evaluation, visualization, CroCo
  stereo/flow, DUSt3R global alignment and the other modules that extraction
  does not use are not included.
- `dust3r/datasets/__init__.py`: the line
  `from f3r.data.components.spann3r_datasets import *` is removed, and
  `data/components/spann3r_datasets/` is not shipped. Those five dataset
  loaders (DTU, 7-Scenes, NRGBD, BlendedMVS and their base class) are Fast3R's
  copies of code from [Spann3R](https://github.com/HengyiWang/spann3r)
  (CC BY-NC-SA 4.0). Feature extraction never uses them.
- `croco/models/curope/` (the CUDA RoPE extension) is intentionally absent. The
  Fast3R README says not to install cuRoPE, because it corrupts Fast3R
  predictions. `croco/models/pos_embed.py` falls back to the PyTorch `RoPE2D`.
