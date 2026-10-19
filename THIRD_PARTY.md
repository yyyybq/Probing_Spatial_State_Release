# Third-party code, model weights and data

The root [`LICENSE`](LICENSE) (CC BY-NC 4.0, inherited from VidFM3D) covers the
code written for this project. Code taken from other projects keeps its own
license. The table below lists that code, and the license files shipped next to
it. The license facts were checked against the upstream license files on
2026-10-19.

## Code

| Component | Location in repo | Upstream | License | Copyright | Local modifications | License file shipped |
| --- | --- | --- | --- | --- | --- | --- |
| VidFM3D (project base) | `vidfm3d/` except the third-party files listed below, among them `vidfm3d/models/components/{dpt_head,backbone_pixalign,probe_pixalign,movement_head}.py` and `vidfm3d/utils/metrics.py` (`vidfm3d/vggt/layers/modulation.py`, which VidFM3D added, is VidFM3D code); `configs/`. In `vidfm3d/train.py` and the `configs/` files adapted from lightning-hydra-template, the code added by VidFM3D and this repository is covered here and the rest is template code (MIT, see its row). Also the VidFM3D-derived wrappers in `features/wan/`, `features/cogvideox/`, `features/aether/aether_feature.py` and `features/{wan,cogvideox,aether,dino,f3r}/extract_features.py` | https://github.com/zxhuang1698/VidFM3D | CC BY-NC 4.0 | Copyright (c) 2025 Zixuan Huang, Xiang Li, Zhaoyang Lv, James M. Rehg | Adapted for the probing protocol of this project (see [README.md](README.md)) | [`LICENSE`](LICENSE) |
| Fast3R | `features/f3r/f3r/` (52 Python modules) | https://github.com/facebookresearch/fast3r, via the VidFM3D copy | FAIR Noncommercial Research License, Last Updated: March 6, 2025 | Copyright (c) Meta Platforms, Inc. and affiliates | Package renamed `fast3r` to `f3r`; inference returns predictions only; `Fast3R.forward` returns per-layer decoder features; `dust3r/utils/image.py` uses a fixed 512x288 LANCZOS resize; optional `random_generator` threaded through `dust3r/inference_multiview.py` and `models/fast3r.py`; pruned to the extraction import closure, without `croco/models/curope`; `dust3r/datasets/__init__.py` no longer imports Fast3R's copies of the Spann3R dataset loaders (`f3r.data.components.spann3r_datasets`, from https://github.com/HengyiWang/spann3r, CC BY-NC-SA 4.0), and those files are not shipped. See [`features/f3r/MODIFICATIONS.md`](features/f3r/MODIFICATIONS.md) | [`features/f3r/LICENSE`](features/f3r/LICENSE) |
| Llama 2 model code (part of Fast3R) | `features/f3r/f3r/models/components/llama.py` | Fast3R copy of `examples/pytorch/tensor_parallel/model.py` from https://github.com/Lightning-AI/pytorch-lightning | The in-file notice, kept unchanged, states a BSD-style license and that "Llama 2 is licensed under the LLAMA 2 Community License" (https://github.com/meta-llama/llama/blob/main/LICENSE). Only model code is included; no Llama weights are used or shipped | Copyright (c) Meta Platforms, Inc. and affiliates | Reformatted only | None separately; see the notice in the file header |
| DUSt3R | `features/f3r/f3r/dust3r/`; `vidfm3d/dust3r/` (`__init__.py`, `datasets/__init__.py`, `datasets/base/{__init__,easy_dataset,batched_sampler}.py`); `vidfm3d/data/components/video_probe_dataset.py` | https://github.com/naver/dust3r, via Fast3R and VidFM3D | CC BY-NC-SA 4.0. ShareAlike applies to these adapted files. Both `dust3r/` copies come from Fast3R and also carry its Meta notice, so the Fast3R license applies to them too | Copyright (c) 2024-present Naver Corporation | Import paths renamed (`f3r.dust3r`, `vidfm3d.dust3r`); Fast3R multi-view changes; `video_probe_dataset.py` adapted for the probe datasets of this project; for `features/f3r/f3r/dust3r/` see [`features/f3r/MODIFICATIONS.md`](features/f3r/MODIFICATIONS.md) | [`features/f3r/f3r/dust3r/LICENSE`](features/f3r/f3r/dust3r/LICENSE) and [`NOTICE`](features/f3r/f3r/dust3r/NOTICE); [`vidfm3d/dust3r/LICENSE`](vidfm3d/dust3r/LICENSE) (also covers `video_probe_dataset.py`) and [`NOTICE`](vidfm3d/dust3r/NOTICE) |
| CroCo | `features/f3r/f3r/croco/` (`models/{blocks,dpt_block,pos_embed}.py`) | https://github.com/naver/croco (branch `master`), via Fast3R | CC BY-NC-SA 4.0. The CroCo `LICENSE` and `NOTICE` add notices for `models/pos_embed.py` (from MAE, CC BY-NC 4.0; it incorporates parts of TensorFlow Models, MoCo v3 and DeiT) and `models/blocks.py` (from timm, Apache-2.0) | Copyright (c) 2022-present Naver Corporation | Import paths renamed (`f3r.croco`); only the 3 model files used by Fast3R are kept; `models/curope/` is omitted | [`features/f3r/f3r/croco/LICENSE`](features/f3r/f3r/croco/LICENSE) and [`NOTICE`](features/f3r/f3r/croco/NOTICE) (match upstream apart from whitespace) |
| VGGT | `vidfm3d/vggt/heads/`, `vidfm3d/vggt/utils/`, `vidfm3d/vggt/layers/__init__.py` | https://github.com/facebookresearch/vggt, via the VidFM3D copy | VGGT License v1, Last Updated: July 29, 2025, including its Acceptable Use Policy | Copyright (c) Meta Platforms, Inc. and affiliates | Identical to the VidFM3D copy: imports rewritten to `vidfm3d.vggt`, reformatted, and changed by VidFM3D. The vendored VGGT revision is not identified (upstream VGGT was CC BY-NC 4.0 until it moved to the VGGT License on 2025-07-29); `LICENSE.txt` is the current upstream text | [`vidfm3d/vggt/LICENSE.txt`](vidfm3d/vggt/LICENSE.txt) |
| VGGT (adapted probe components) | `vidfm3d/models/components/dpt_head.py`; `vidfm3d/models/components/backbone_pixalign.py`; `vidfm3d/models/components/probe_pixalign.py`; `vidfm3d/models/components/movement_head.py` | https://github.com/facebookresearch/vggt: `vggt/heads/dpt_head.py`, `vggt/models/aggregator.py`, `vggt/models/vggt.py` and `CameraHead` in `vggt/heads/camera_head.py`; the first three via VidFM3D | As the VGGT row above | Copyright (c) Meta Platforms, Inc. and affiliates | VidFM3D adapted the first three files to video-feature tokens (`dpt_head.py`: absolute imports, other default channels and layer indices, `n_patches`/`frames_shape` inputs instead of images; `backbone_pixalign.py`: the alternating-attention aggregator over projected video features; `probe_pixalign.py`: the VGGT model structure and forward docstring); VidFM3D's copies carry no Meta header. This repository changed one line of `dpt_head.py` (`view` to `reshape`), added the movement and identity heads to `probe_pixalign.py`, left `backbone_pixalign.py` unchanged, and wrote `movement_head.py` after `CameraHead` (same layers and refinement loop, on adjacent camera-token differences). Each file now has a header naming VGGT, Meta and the license. The "Inspired by Depth-Anything-V2" line in `dpt_head.py` is VGGT's own | [`vidfm3d/vggt/LICENSE.txt`](vidfm3d/vggt/LICENSE.txt) |
| DINOv2 layers (part of VGGT) | `vidfm3d/vggt/layers/{attention,block,drop_path,layer_scale,mlp,patch_embed,rope,swiglu_ffn}.py` (the files whose headers state Apache-2.0) | https://github.com/facebookresearch/dinov2, via VGGT and VidFM3D | Apache-2.0 | Copyright (c) Meta Platforms, Inc. and affiliates | Imports rewritten, reformatted | [`vidfm3d/vggt/layers/LICENSE`](vidfm3d/vggt/layers/LICENSE) |
| VGGT and PoseDiffusion evaluation helpers | `vidfm3d/utils/metrics.py` | https://github.com/facebookresearch/vggt (branch `evaluation`, `evaluation/test_co3d.py`: `build_pair_index`, `rotation_angle`, `translation_angle`, `compare_translation_by_angle`, `calculate_auc_np`, `se3_to_relative_pose_error`), which builds on https://github.com/facebookresearch/PoseDiffusion (`pose_diffusion/util/metric.py`: `calculate_auc`, `batched_all_pairs`, `compare_translation_by_angle`); via VidFM3D | CC BY-NC 4.0 for both parts (the same license as the root `LICENSE`): PoseDiffusion is CC BY-NC 4.0, and the VGGT `evaluation` branch, where `test_co3d.py` lives, still carries the original CC BY-NC 4.0 license | Copyright (c) Meta Platforms, Inc. and affiliates | Changed by VidFM3D: imports rewritten to `vidfm3d.vggt`, `calculate_auc` gained `return_list`, `batched_se3_to_relative_pose_error` added. This repository only added the file header, which names both sources | [`LICENSE`](LICENSE) |
| Aether | `features/aether/aether/`; `OneStepAetherPipeline` in `features/aether/aether_feature.py` adapts `AetherV1PipelineCogVideoX` | https://github.com/InternRobotics/Aether (formerly OpenRobotLab/Aether), via the VidFM3D copy | MIT | Copyright (c) 2025 Aether Team | Relative import at `pipelines/aetherv1_pipeline_cogvideox.py` line 22; `resize_mode="default"` at line 495; upstream `utils/postprocess_utils.py` lines 119-122 removed; `utils/visualize_utils.py` not included | [`features/aether/LICENSE`](features/aether/LICENSE) (with a note listing these changes) |
| Hugging Face diffusers (adapted code) | `features/wan/wan_feature.py`, `features/wan/wan_feature_i2v.py`, `features/cogvideox/cogvideox_feature.py`, `features/cogvideox/cogvideox_feature_i2v.py`, `features/aether/aether_feature.py`; functions marked "Copied from diffusers" in `features/aether/aether/pipelines/aetherv1_pipeline_cogvideox.py` | https://github.com/huggingface/diffusers | Apache-2.0 | Copyright 2025 The Wan Team and The HuggingFace Team (Wan files); Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team (CogVideoX and Aether files) | Pipelines and transformer forward adapted for one-step feature extraction: the clip is noised to a single timestep and hidden states are returned from selected transformer blocks | Apache-2.0 header in each file; the full Apache-2.0 text is also in [`vidfm3d/vggt/layers/LICENSE`](vidfm3d/vggt/layers/LICENSE) |
| lightning-hydra-template | `vidfm3d/train.py` (the structure and most lines of the template's `src/train.py`); `vidfm3d/utils/{instantiators,logging_utils,pylogger,rich_utils,utils}.py`; `features/f3r/f3r/utils/{instantiators,logging_utils,pylogger,rich_utils,utils}.py`; `configs/__init__.py`, `configs/train.yaml`, `configs/{hydra,extras,paths,trainer,logger}/` and `configs/callbacks/{default,early_stopping,model_checkpoint,model_summary,rich_progress_bar}.yaml`; the layout of `configs/` | https://github.com/ashleve/lightning-hydra-template (the `features/f3r/f3r/utils/` copies via Fast3R; the rest via VidFM3D) | MIT | Copyright (c) 2021 ashleve | Adapted by VidFM3D and Fast3R. This repository adds the probe world-size guard, the run manifest and a ModelCheckpoint save-last patch to `vidfm3d/train.py`, and new keys to `configs/train.yaml` and `configs/callbacks/default.yaml`. `configs/__init__.py` holds only the template's one-line comment; its Meta Platforms header was added by Fast3R and kept by VidFM3D, and no Meta code is in that file | Full text [below](#lightning-hydra-template-mit-license) (upstream keeps it in the README) |
| PyTorch3D | `vidfm3d/vggt/utils/rotation.py` ("Modified from PyTorch3D"); `_handle_pointcloud_input` in `vidfm3d/utils/eval_utils.py` (from `pytorch3d/loss/chamfer.py`) | https://github.com/facebookresearch/pytorch3d | BSD License (3-clause) | Copyright (c) Meta Platforms, Inc. and affiliates | Adapted by VGGT and VidFM3D | Full text [below](#pytorch3d-bsd-license) |
| Depth-Anything-V2 | `vidfm3d/models/components/dpt_head.py` ("Inspired by" line) | https://github.com/DepthAnything/Depth-Anything-V2 | Apache-2.0 (code) | - | Acknowledgement only. The "Inspired by" line is VGGT's own: it came with VGGT's DPT head (`vggt/heads/dpt_head.py`), from which this file is adapted (see the VGGT (adapted probe components) row) | None |

## Model weights

The code downloads or loads these checkpoints; this repository does not
redistribute any weights. Each checkpoint is governed by its own license and
model card. The licenses below are taken from the upstream code repositories;
check the model card before use.

| Checkpoint | Backend | License |
| --- | --- | --- |
| `facebook/dinov2-large` | `dino` | Apache-2.0 (facebookresearch/dinov2) |
| `facebook/vjepa2-vitl-fpc64-256` | `vjepa2` | V-JEPA 2 repository (facebookresearch/vjepa2) is MIT; follow the model card |
| `MCG-NJU/videomae-large` | `videomae` | CC BY-NC 4.0 (MCG-NJU/VideoMAE) |
| `jedyang97/Fast3R_ViT_Large_512` | `f3r` | FAIR Noncommercial Research License (facebookresearch/fast3r) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2_5_vl` (`features/run_inscene15k_mllm.py`) | Apache-2.0 (QwenLM/Qwen2.5-VL) |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `wan` | Apache-2.0 (Wan-Video/Wan2.1) |
| `THUDM/CogVideoX-5b-I2V` | `cogvideox`; also the tokenizer, text encoder, VAE and scheduler for `aether` | The CogVideoX License (`MODEL_LICENSE` in THUDM/CogVideo, now zai-org/CogVideo) |
| `AetherWorldModel/AetherV1` | `aether` (transformer) | See the model card |
| `MiniMaxAI/MiniMax-H3` | `minimax_h3` | MiniMax H3 Community License |
| `robbyant/lingbot-video-moe-30b-a3b`, with the source code of https://github.com/Robbyant/lingbot-video | `lingbot` | Apache-2.0 (Robbyant/lingbot-video) |

Other checkpoints that the code can load (for example `THUDM/CogVideoX-2b`,
`Qwen/Qwen2.5-VL-3B-Instruct` or the larger V-JEPA 2 variants) follow their own
model cards.

## Dataset

The experiments use `lifuguan/InsScene-15K`, a ScanNet++ subset. Its use is
governed by the InsScene-15K terms and the ScanNet++ terms of use. This
repository ships only scene ids, frame filenames and window indices in
`data/splits/`; it does not ship images, depth, poses or other data.

## License texts not shipped elsewhere

### lightning-hydra-template (MIT License)

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### PyTorch3D (BSD License)

```
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The repository as a whole is for non-commercial research use.
