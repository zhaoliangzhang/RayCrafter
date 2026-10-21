# RayCrafter

Multi-view refinement and novel-view synthesis on a pretrained image
diffusion transformer. Given R clean reference images with cameras and V
target cameras, the model produces the V target images. Each target starts
from a degraded render (refinement) or from pure noise (novel-view synthesis).

The goal is a model that does not depend on one backbone or one camera type:
backbone code and camera encodings are kept in separate modules. Today there is
one backbone (FLUX.2 [klein]) and two camera encodings, SCoPE and RayRoPE, for
pinhole cameras.

- **Bridge matching.** `z_t = t·z_src + (1-t)·z_tgt`, where `t=1` is the
  source and `t=0` is clean. The model predicts the velocity `z_src - z_tgt`
  on the target-view tokens. Refs and targets go in as one token sequence
  `[ref_1 … ref_R | view_1 … view_V]`, and the backbone itself is not changed.
- **SCoPE** (`camera_condition: scope`). For each token, the Plücker ray
  through its patch is added to attention Q/K, plus a per-frame residual on V.
- **RayRoPE** (`rope_type: rayrope`). 96 of the 128 RoPE channels in each head
  are replaced by projective ray coordinates in each query camera's frame,
  using a per-token predicted depth.
- All camera modules start at zero. At step 0 the model is exactly the
  pretrained backbone.

## Setup

```bash
git clone --recurse-submodules <this repo> RayCrafter && cd RayCrafter
# (already cloned without it: git submodule update --init)

# FLUX.2 / FLUX 3 model code are the submodules third_party/flux2 and
# third_party/flux3, pinned to commits. Their own deps are pinned too
# tightly, hence --no-deps.
pip install --no-deps -e third_party/flux2
pip install --no-deps -e third_party/flux3   # only for backbone: flux3

pip install -e .            # torch>=2.4 with CUDA should already be installed
huggingface-cli login       # FLUX.2-klein-4B and FLUX.2-dev (VAE) are gated on HF
```

If flux2 / flux3 are not pip-installed, `third_party/<name>/src` is also found
automatically. Weights download on first use. To use local files, set
`model_path` / `ae_path` in the config.

## Train

```bash
cp configs/example.yaml configs/my_run.yaml     # edit train_shards / val_shards / output_dir
python train.py --config configs/my_run.yaml                                      # 1 GPU
torchrun --standalone --nproc_per_node=2 train.py --config configs/my_run.yaml    # DDP
```

`configs/example.yaml` explains every field. It is our FLUX.2-klein-4B +
SCoPE + RayRoPE recipe: full fine-tune, 1 target + 6 refs at 432×768, 2×H200
for 48 h. Unknown config keys raise an error. `output_dir` gets `config.yaml`
and `transformer_step<N>.safetensors` / `transformer_final.safetensors` (bf16).

## Sample

```bash
python sample.py --config outputs/<run>/config.yaml \
    --ckpt outputs/<run>/transformer_final.safetensors \
    --mode nvs --num_steps 4 --max_batches 8        # or --mode refine; --shards to override val_shards
```

## Data format

WebDataset tars. Each sample is one frame:

| file | content |
|---|---|
| `{key}.png` | degraded render of the frame (model input for refinement) |
| `{key}.gt.png` | clean frame (target) |
| `{key}.ref.png` | a clean reference image of the same scene |
| `{key}.pose.json` | cameras of the frame and of its ref |

`key` is `{a}__{b}__{c}__{d}__{..._frameindex}`: at least 5 fields joined by
`__`. All fields except the last form the sequence id, and the last field ends
in `_<int>` (the frame index). Samples from one sequence must be contiguous in
the tar. When a sample is built, the target views come from frames of the same
sequence (within `±window`), and the R refs are distinct `ref.png`s of that
whole sequence (the loader reads a sequence completely before building its
samples).

`pose.json`, either layout:

```jsonc
// flat: intrinsics normalized by image width/height
{"camera_c2w": [[4x4]], "camera_fxfycxcy": [fx/W, fy/H, cx/W, cy/H],
 "ref_camera_c2w": [[4x4]], "ref_camera_fxfycxcy": [...]}
// colmap-style: intrinsics in pixels of a width×height image
{"view": {"c2w": [[4x4]], "camera": {"width": W, "height": H, "params": [fx, fy, cx, cy, ...]}},
 "ref":  {"c2w": [[4x4]], "camera": {...}}}
```

`c2w` is camera-to-world in the OpenCV convention (x right, y down, z forward).
Any world frame works, because poses are re-expressed relative to camera 0.
The curriculum groups tars by a `dsim_*` directory in their path and draws a
bucket for every sample from the probabilities at the current step (see
`configs/example.yaml`).

## Code layout

```
train.py, sample.py            entry points
configs/example.yaml           the only config; copy it
src/raycrafter/
  config.py                    dataclass mirror of example.yaml
  model.py                     bridge matching: training_step / sample (backbone-generic)
  lora.py                      LoRA + checkpoint save/load
  backbones/flux2.py           everything FLUX.2-specific: loading, VAE, token packing, block patches
  backbones/flux3.py           FLUX 3 Image Klein: loading, token packing, batched camera-aware forward
  cameras/scope.py             SCoPE: Plücker rays + Q/K/V injection (backbone-free)
  cameras/rayrope.py           RayRoPE: projective geometry + attention (backbone-free)
  data/                        WebDataset loader, multi-view grouping, curriculum
```

**New backbone:** add `backbones/<name>.py` with the same functions as
`flux2.py` (`load_backbone`, `load_vae`, `pack_views`, `unpack_views`,
`ref_index`, `run_transformer`, `install_camera_attention`, `camera_inputs`,
`set_camera_context`, ...). Then add it to `BACKBONES` in `model.py` and to
the `backbone` choices in `config.py`. Two places in `cameras/` assume FLUX.2: RayRoPE's 32 + 96
head-channel split (`ROPE_DIM`, `RAY_DIM`), and the 16-px token patch.

**New camera type:** cameras enter in exactly two functions.
`scope.plucker_tokens` builds one ray per token, and
`rayrope.prepare_geometry` / `_query_positions` build projection matrices.
Both assume pinhole `fxfycxcy`. A generic camera (fisheye, panorama, ...)
needs a per-token ray (origin, direction) from those two functions. The rest
of the pipeline works on rays and is unchanged.

## Relation to geometric_syncfix

This is the FLUX + SCoPE + RayRoPE path of `geometric_syncfix`, minus the LBM/SDXL
code, ray-image layouts (rothko / checkerboard), and CameraAnything.
The outputs and gradients of `training_step` and `sample` match the old code
bit-for-bit. Old `transformer_*.safetensors` from scope / rayrope runs load
directly (`resume_from`). The config keys `camera_conditioning`,
`raymap_layout`, `use_pose`, `text_prompt`, `guidance_scale` are gone, and
`resume_lora` is now `resume_from`. Two fixes over the old code: LoRA
checkpoints now include the camera-module weights, and LPIPS stays in eval mode
during training (it was running with Dropout(0.5)).

The data loader differs on purpose. The old one drew the curriculum bucket and
noise level once per tar, and built each target's refs only from frames streamed
before it. With 5000-sample tars, the realized mix stayed near the curriculum's
start probabilities for the whole run, and most samples got repeated refs. Now
the bucket and noise level are drawn per sample, targets are random frames of
whole buffered sequences, and refs come from the whole sequence.
