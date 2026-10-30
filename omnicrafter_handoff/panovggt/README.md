# PanoVGGT Two Reference Point Cloud Rendering

Generate noisy panoramic point-cloud renders for OmniCrafter cleanup inputs from
two clean reference panoramas and known camera poses. This is a portable export
of the pipeline used for our HM3D, Replica, and JHU evaluation priors.

The pipeline is:

```text
2 reference ERP RGB images
    -> PanoVGGT camera poses and colored world points
    -> similarity alignment using only the 2 reference GT poses
    -> raw ERP point rendering at each target GT pose
    -> prior PNGs, validity masks, alignment diagnostics
```

Target RGB and target depth are never inputs to reconstruction, alignment, or
rendering. Target RGB paths are optional metadata and are not opened or checked.
No OmniCrafter checkpoint, diffusion model, training pipeline, dataset images,
or PanoVGGT weights are included.

## Files

| File | Purpose |
| --- | --- |
| `build_manifest.py` | Select two reference frames and one or more target frames from a posed sequence |
| `render_priors.py` | Validate inputs, run PanoVGGT, align, render, and resume completed samples |
| `panovggt_point_predictor.py` | Ordered reference preprocessing and full PanoVGGT inference |
| `geometry.py` | Original similarity alignment and raw ERP rasterizer |
| `calibrate_scale.py` | Optional fallback-scale calibration for co-located reference cameras |
| `configs/panovggt.yaml` | Inference configuration matching the original model settings |
| `examples/sequence.json` | Editable input template with toy poses, not actual dataset data |
| `vendor/PanoVGGT/` | Pinned PanoVGGT Python source and upstream license |
| `provenance/original/` | Unmodified snapshots of our original pipeline scripts |
| `tests/test_pipeline.py` | CPU geometry and synthetic end-to-end regression tests |

## Environment and checkpoint

Use Linux and Python 3.11 with a CUDA-enabled PyTorch installation. The original
pipeline uses bfloat16 autocast on CUDA; use a GPU supporting bfloat16. There is
no multi-GPU requirement: one process handles one reference pair at a time.
Memory requirements depend on the GPU and attention implementation; this package
does not establish a minimum VRAM guarantee.

The observed source environment used PyTorch 2.5.0, torchvision 0.20.0, and
xFormers 0.0.28.post2. An installation matching that stack for CUDA 12.4 is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.0 torchvision==0.20.0 xformers==0.0.28.post2 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Choose a different matching torch/torchvision/xFormers wheel set if your GPU or
driver requires it. `requirements-tested.txt` records the source environment's
key package versions; it is not a full environment lockfile. Open3D, Gradio,
COLMAP, FrameCrafter, and the PanoVGGT training dependencies are not needed by
this entry point. The upstream Python model package is included for provenance,
but unused utilities may have additional dependencies.

The optional compiled RoPE2D extension is not included. The warning about using
the PyTorch RoPE2D fallback is expected; it can affect speed, not the requested
pose convention or rendering pipeline.

Download the full checkpoint from the official
[PanoVGGT weights](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt):

```bash
mkdir -p checkpoints
curl -fL --retry 3 \
  https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt \
  -o checkpoints/model.pt
sha256sum checkpoints/model.pt
```

The checkpoint used for our evaluation had:

```text
size:   3939890032 bytes
SHA256: 4adab888064ef206c20ab42c08c5f973e3bd98a813fa2c3d849b4d445e278670
```

Loading should report **1293 tensors, 0 missing, 0 unexpected**. The portable
wrapper rejects missing or unexpected keys instead of silently using unmatched
weights. `torch.load(..., weights_only=False)` follows the original loader;
only load a checkpoint from a trusted source.

The bundled upstream constructor also attempts to initialize DINOv2 through
`torch.hub`, with a direct-download fallback. The first model construction may
therefore need internet access and an additional cache download. Its weights are
subsequently overwritten by the full PanoVGGT checkpoint. The model source is
not modified to bypass this initialization.

## Prepare images and poses

Each sample needs exactly two clean, full 360-degree RGB panoramas in 2:1
equirectangular projection, their two reference poses, and one or more target
poses. Reference files may be PNG or JPEG and need not be pre-downscaled.

All poses must be 4 by 4 OpenCV **world-to-camera** matrices:

```text
p_camera = R_w2c @ p_world + t_w2c
x points right, y points down, z points forward
```

All views in a sample must share the same world coordinate frame and units.
If poses are camera-to-world, invert them first. If they use OpenGL or another
axis convention, convert both reference and target poses consistently before
running this code. Merely inverting an OpenGL pose is not an axis conversion.

ERP longitude zero is at the horizontal center looking along camera +z. The
horizontal seam is toward -z; the top is -y. Do not horizontally roll, mirror,
crop, or rotate the images without updating the associated orientation.

### Select reference and target frames

Edit `examples/sequence.json` to point at your own reference images and real
poses. Its toy trajectory has reference frames 0 and 4 and target frames 1, 2,
and 3. Relative image paths are resolved against the sequence JSON's directory.

```bash
python build_manifest.py \
  --sequence examples/sequence.json \
  --references 0 4 --targets 1 2 3 \
  --height 256 --width 512 --output benchmark
```

Selection is explicit, not random. The generator does not choose reference
images based on target appearance. Choose a meaningful nonzero reference
baseline with overlapping visible content. For a one-target sample, use
`--targets 2`; for ten targets, provide ten indices from your sequence.

The builder creates one sample per invocation and refuses to replace an existing
manifest. For large datasets, write a manifest containing multiple samples in
the following format. It also accepts the existing evaluation benchmark schema.

### Benchmark manifest

`--benchmark` points to a directory containing `manifest.json`:

```json
{
  "pose_convention": "OpenCV world-to-camera",
  "resolution": {"height": 256, "width": 512},
  "logical_sets": 1,
  "target_images": 1,
  "sets": [
    {
      "id": "sample_000000",
      "scene_name": "my_scene",
      "source_sample": "my_pair",
      "references": [
        {"frame_index": 0, "image": "../data/ref0.png",
         "w2c": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        {"frame_index": 4, "image": "../data/ref4.png",
         "w2c": [[1,0,0,-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
      ],
      "targets": [
        {"target_number": 0, "frame_index": 2,
         "w2c": [[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
      ]
    }
  ]
}
```

Sample IDs must be contiguous: `sample_000000`, `sample_000001`, and so on.
Target numbers are zero-based and contiguous within each sample. Set
`logical_sets` to the number of samples and `target_images` to the total number
of target poses. Each view needs an integer `frame_index`. A target may include
an `image` field for later scoring; this renderer does not use it. Relative
paths in a benchmark manifest resolve against the manifest's directory.

To reproduce an existing evaluation, retain its exact reference pairs, target
order, and poses. Re-selecting frames changes the benchmark.

## Validate and render

Validate without loading a checkpoint or using a GPU:

```bash
python render_priors.py --benchmark benchmark --validate-only
```

Render all samples at height 256 and width 512:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_render.sh \
  --benchmark benchmark --output outputs/priors \
  --checkpoint checkpoints/model.pt --height 256 --width 512
```

Use `--max-sets 1` for a first-sample smoke test, then rerun without that limit.
`PYTHON_BIN=/path/to/python` can select the interpreter for `run_render.sh`.
Use `--config` to override the included config only when deliberately changing
the model. Use a new output directory if the checkpoint, manifest, code, or
resolution changes; the run contract prevents incompatible results from being
silently mixed.

Run the same command again to resume. Completed samples are checked against
reference image hashes and poses; incomplete or changed samples are recomputed.
`--overwrite` recomputes assigned samples within the same run contract. Never
run duplicate workers for the same shard and output directory.

### Multiple GPUs

Each worker loads its own PanoVGGT model and handles a disjoint set of samples.
For four GPUs, run this from the package directory:

```bash
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python -u render_priors.py \
    --benchmark benchmark --output outputs/priors \
    --checkpoint checkpoints/model.pt --height 256 --width 512 \
    --num-shards 4 --shard-index "$gpu" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
```

Use only GPUs allocated to you. Sharding is by original sample index modulo the
number of shards. There is no distributed training, SSH, scheduler, or automatic
GPU selection in this package.

## Resolution and alignment details

PanoVGGT always resizes the ordered references to **height 518 by width 1036**,
using PIL Lanczos and RGB values in [0,1]. Output rendering is separate:
`--height 256 --width 512` produces a 256 by 512 prior directly, rather than
resizing a rendered image. Increasing output resolution does not increase the
PanoVGGT reconstruction resolution. The full model runs in evaluation mode;
camera, point, depth, and global-point branches retain the original settings.

For predicted and GT reference camera-to-world rotations `Rp_i` and `Rg_i`,
the alignment rotation is the closest proper rotation to
`sum_i(Rg_i @ Rp_i.T)`, obtained by SVD. Scale is the ratio of GT and predicted
reference-center distances. Translation is the mean residual of the two aligned
reference centers. The transformation applied to every predicted point is:

```text
p_gt_world = scale * rotation @ p_pred_world + translation
```

This is reference-pose similarity alignment, not ICP, target-image fitting, or
an unconstrained pose optimization. Metadata records both reference-center and
orientation residuals. Two centers alone cannot constrain every rotation, so
the known reference orientations are essential here.

At a target pose, points are transformed into its camera frame. Longitude is
`atan2(x,z)` and downward latitude is `asin(y/radius)`. A point lands in one ERP
pixel; the nearest **radial depth** wins when points collide. Empty pixels stay
black. There is no hole filling, Gaussian splatting, confidence filtering,
target-depth filtering, or online pixel-drop augmentation in this stage. Those
additional corruptions belong to downstream training, not prior generation.

### Co-located reference cameras

A reference-center distance below `1e-4` in your pose units is treated as
degenerate. This pair cannot determine metric scale. The portable entry point
rejects it unless you explicitly provide a calibrated fallback; the original
driver could produce an uncalibrated first-pass result before repair.

The previous Replica pipeline used the median scale from non-degenerate
reference pairs. To apply that same fallback policy:

```bash
python render_priors.py --benchmark benchmark --output outputs/priors \
  --checkpoint checkpoints/model.pt --skip-degenerate-baselines
python calibrate_scale.py --benchmark benchmark --priors outputs/priors \
  --output outputs/scale_calibration.json
python render_priors.py --benchmark benchmark --output outputs/priors \
  --checkpoint checkpoints/model.pt --only-degenerate-baselines \
  --fallback-scale-calibration outputs/scale_calibration.json
```

This is an explicit scale prior, not scale recovered from the degenerate pair.
Calibration uses reference-pose alignment metadata only, never target RGB or
depth. Use consistent pose units and an appropriate calibration population;
do not transfer an arbitrary median across unrelated datasets. If no suitable
non-degenerate pairs exist, use another reference pair or a justified external
scale calibration. A zero predicted reference baseline is rejected even when
a GT-baseline fallback is provided.

## Outputs

```text
outputs/priors/
  run_contract.json
  configuration_shard_00.json
  progress_shard_00.jsonl
  _SUCCESS.shard00
  samples/sample_000000/
    prior_0.png
    mask_0.png
    prior_1.png
    mask_1.png
    prior_2.png
    mask_2.png
    metadata.json
    _SUCCESS
```

`prior_N.png` is an RGB point-cloud render. `mask_N.png` is 255 where a point
was rasterized and 0 elsewhere. A black RGB value alone does not distinguish a
hole from a genuinely black surface; use the mask. Metadata records the input
references, alignment scale and transform, residuals, valid point count, pixel
coverage, and elapsed time. The point cloud and radial depth arrays are computed
but not saved by default, matching the prior-generation workflow.

One PanoVGGT forward pass reconstructs a sample's two references, then all its
target poses are rendered from that cloud. Separate sample rows are processed
independently, even if they happen to use the same reference pair. There are no
PSNR, SSIM, LPIPS, or DreamSim scores in this package; scoring requires a separate
GT-aware evaluation stage.

## Tests and provenance

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

The tests cover known similarity transforms, exact numerical agreement with the
original geometry functions, nearest-depth visibility, holes, manifest selection,
pose validation, relative paths, resume identity, degenerate baselines, and a
synthetic end-to-end run with a fake predictor and nonexistent target RGB paths.
They do not measure real-model accuracy or GPU memory usage. The export has not
been subjected to a new full-checkpoint GPU inference run.

Upstream PanoVGGT is pinned to commit
`18575370eb6fdf447342a055701193d20cd859e6` from
[YijingGuo-June/PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT).
`provenance/source_manifest.json` records exported file hashes and source script
locations relative to our original workspace. The original production files
and datasets were not modified when creating this package.
Vendored model source has identical Python syntax trees to the source snapshot;
the text export normalizes missing final newlines. Original local-script
snapshots are byte-identical. All 14 CPU tests passed during packaging.

The numerical alignment/rasterization functions and model inference settings
are preserved. Portability changes add relative paths, optional target RGB
metadata, manifest validation, safer resume checks, strict checkpoint matching,
and explicit rejection/calibration of degenerate reference baselines. The
original script snapshots are included to make those changes inspectable.

PanoVGGT's MIT license is in `vendor/PanoVGGT/LICENSE`. Files derived from
Meta's DINOv2 retain their copyright notices and Apache 2.0 headers; the Apache
license text is included at `vendor/LICENSE-Apache-2.0`. These notices apply to
the respective third-party code, not automatically to all local wrapper code
or downloaded model weights. Consult the upstream model's terms when sharing
weights. Cite PanoVGGT when using its reconstruction model:

```bibtex
@article{guo2026panovggt,
  title={PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery},
  author={Guo, Yijing and Chao, Mengjun and Wang, Luo and Zhao, Tianyang and
          Dai, Haizhao and Zhang, Yingliang and Yu, Jingyi and Shi, Yujiao},
  journal={arXiv preprint arXiv:2603.17571},
  year={2026}
}
```
