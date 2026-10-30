# OmniCrafter: panorama NVS and point-cloud-render cleanup

Research-code sharing snapshot, 2026-10-19. OmniCrafter is the model previously
called **FrameCrafter Model 2**. This package preserves the local model/training
implementation used by our experiments and adds portable launchers and examples.
It is **not** the unmodified upstream FrameCrafter release, the SyncFix model,
FixAnything, or the proposed SCoPE variant.

## What the model does

The standard setting is **two clean reference panoramas + three query slots**.
Every slot has a camera pose. Each query condition can be:

- an exactly black RGB image: novel-view synthesis (NVS);
- an aligned noisy point-cloud render: generative cleanup.

The output is a clean RGB panorama for each query pose. A sample may contain both
query types. Ground-truth target RGB is used for training/scoring, never for
inference conditioning. No depth map is supplied to this RGB variant.

The backbone is **Wan2.1-I2V-14B-480P**. Frames are independently VAE-encoded;
the temporal layout is `[reference_0, reference_1, query_0, query_1, query_2]`.
All cameras are normalized together: the **last query camera** becomes the origin,
then translations are divided by the mean camera-center distance. ERP rays use
OpenCV coordinates (right, down, forward) and Plücker features `[d, o × d]`.
PixelUnshuffle(8) packs six ray channels into 384 channels. These are concatenated
with the 36 WAN latent/conditioning channels, giving **420 DiT input channels**.
This is channel concatenation before patch embedding, not extra camera tokens.
Temporal RoPE is neutralized in the local panorama path; spatial RoPE remains.

Training uses target-only flow-matching supervision, rank-32 DiT LoRA on
`q,k,v,o,ffn.0,ffn.2`, and a fully trainable 420-channel patch embedding.
The remaining backbone weights are frozen. The text encoder is retained even
though the standard prompt is empty. Checkpoints contain adapter/patch weights,
not a standalone copy of the whole WAN backbone.

## Contents

```text
code/                         copied model and DiffSynth source
  model_pose_conditioned_render_queries.py   OmniCrafter inference API
  model.py                                  WAN/adapter loading
  model_training/train_pose_conditioned_render_queries.py
  model_training/train.py                   shared training module
  diffsynth/core/data/pose_conditioned_render_query_dataset.py
  diffsynth/core/data/dataset.py             ERP rays and pose normalization
  diffsynth/pipelines/wan_video.py           image/query conditioning
  diffsynth/diffusion/                       loss, optimizer loop, checkpointing
configs/accelerate_zero2_gpu.yaml            ZeRO-2, no optimizer/parameter offload
scripts/train.py                            portable multi-GPU launcher
scripts/infer.py                            portable NVS/cleanup inference
scripts/score.py                            PSNR, SSIM, WS-PSNR; optional LPIPS/DreamSim
scripts/check_dataset.py                    CPU data preflight
scripts/make_toy_data.py                     synthetic format fixtures
examples/                                   training and pose-manifest examples
tests/                                      CPU regression/smoke tests
provenance/                                 source hashes and historical launchers
```

The copied `diffsynth/` tree includes helper modules for other architectures;
their presence does not mean OmniCrafter uses them. `code/README_FrameCrafter.md`
is retained for attribution/upstream context; follow **this README** for our
panorama input format. Production source files were not edited while packaging.

Not included: datasets, personal validation manifests, trained checkpoints, WAN
weights, optimizer state, environment binaries, experiment outputs, GPU queue
controllers, distillation training, or PanoVGGT itself. This is the standard
multi-step OmniCrafter code, not the separate 20-to-2-step distillation package.

## Setup

Use Linux, Python 3.11, and a CUDA-enabled PyTorch environment. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Example matching our observed torch build; choose a CUDA build supported by your machine.
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements-observed.txt -e ./code
python -m pip install PyYAML
```

`requirements-observed.txt` records the core versions on the packaging machine,
not a guarantee for every GPU/driver. DeepSpeed may compile CUDA extensions and
needs an appropriate CUDA/toolchain installation. CPU tests require no weights:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
```

Do not replace the bundled DiffSynth implementation with an arbitrary upstream
version: the panorama conditioning/training modifications are in this copy.
The portable launchers explicitly put `code/` first on Python's import path.

## Weights to obtain separately

1. WAN backbone files from `Wan-AI/Wan2.1-I2V-14B-480P` and its UMT5 tokenizer.
2. A compatible FrameCrafter initialization for training, or an OmniCrafter
   adapter checkpoint for inference/continuation. Ask the sender for the desired
   checkpoint; it is deliberately not bundled. The 650k teacher checkpoint is
   the file named `step-650000.safetensors` from the 2ref3 mixed-task run.

The `--wan-root` directory must have this layout (not the model directory itself):

```text
wan21/
  Wan-AI/
    Wan2.1-I2V-14B-480P/
      diffusion_pytorch_model-00001-of-00007.safetensors
      ... the other six DiT shards ...
      models_t5_umt5-xxl-enc-bf16.pth
      Wan2.1_VAE.pth
      models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    Wan2.1-T2V-1.3B/google/umt5-xxl/
      tokenizer_config.json
      ... the complete tokenizer files ...
```

The loader also accepts corresponding converted T5/VAE/CLIP `.safetensors` under
`wan21/DiffSynth-Studio/Wan-Series-Converted-Safetensors/` (preferred when present).
Place large weights on local storage if possible. Model loading begins on CPU;
multiple ranks can require substantial system RAM before GPU training begins.
Only load checkpoints from trusted sources, especially the original `.pth` files.

## Training data format

Images are full **2:1 equirectangular RGB panoramas**. The dataloader resizes to
the configured height/width using bilinear interpolation; no pre-downsampling
is necessary. The standard resolution here is **height 256 × width 512**.
Dimensions must be divisible by 16. Upscaling low-resolution data does not restore
missing detail.

Training uses a JSONL file with **one row per target**, so a 2ref3 set occupies
three consecutive lines sharing `source_sample`. See `examples/train.jsonl`.
Each row has:

| Key | Meaning |
| --- | --- |
| `source_sample` | Unique logical set ID; all queries of that set stay together |
| `scene_name` | Key in the pose manifest |
| `target_number` | Contiguous query ordinal 0, 1, 2 |
| `refs` | Same two clean reference paths in every row |
| `src` | Noisy point-cloud render at this target camera |
| `target` | Clean target RGB, for supervision only |
| `frame_index` | Index of the target camera in the scene pose array |

Paths are absolute or relative to `--data-root`. **Reference filenames must have
integer stems**, e.g. `000073.png`, because their pose indices are parsed from the
filename. Target pose indices come from `frame_index`; noisy render filenames do
not have to be numeric. Use valid nonnegative indices. Avoid duplicate sample
groups separated in the JSONL: streaming treats contiguous groups as samples.

The separate pose manifest is illustrated in `examples/pose_manifest.json`:

```json
{"pose_convention":"w2c_opencv","scenes":{
  "demo_scene":{"pose_path":"poses/scene.npz","num_frames":5}
}}
```

Each NPZ contains a float32 `w2c_poses` array of shape `[num_frames,4,4]`.
`pose_path` is relative to `--data-root` (or absolute), **not** relative to the
pose-manifest file. Poses are world-to-camera, OpenCV, using a common scene frame
and consistent translation units. Convert c2w/OpenGL data explicitly first.

The noisy render must already align with the target pose and ERP orientation.
Generate it from the **two references only**, e.g. PanoVGGT → point cloud → align
predicted reference poses/cloud to GT reference poses → render target cameras.
Do not use target RGB/depth to construct inference priors. The separately shared
`panovggt_two_reference_renderer_20261015` package implements this preprocessing.
OmniCrafter does not run PanoVGGT inside its training dataloader.

Keep **all stored training rows pointing to noisy renders**. Black inputs are
created online; do not rewrite the data to change the task mixture. Dataset
balance is determined by the manifest, not by an internal HM3D/JHU sampler.
For a 1:1 mixture, use equal numbers of complete HM3D and JHU groups, excluding
the held-out test scenes. Train and validation manifests must remain separate.

Generate a tiny synthetic example and check its schema:

```bash
python scripts/make_toy_data.py --output toy_data
CUDA_VISIBLE_DEVICES='' python scripts/check_dataset.py \
  --data-root toy_data --train-manifest toy_data/train.jsonl \
  --pose-manifest toy_data/poses/manifest.json --height 32 --width 64
```

These random images are **only fixtures**, not suitable for model training or
performance claims. For real data, use `--max-samples 0` to check every set;
the default preflight loads the first eight. Streaming creates a small byte-offset
index beside the JSONL, or falls back to an in-memory index if it cannot write.

## Task-mixture controls

Two distinct controls exist. Do not enable both:

| Recipe | `--nvs-ratio` | `--nvs-probability` | Behavior |
| --- | ---: | ---: | --- |
| 650k teacher continuation | 0.5 | 0 | Each query independently black with probability 0.5 |
| Sample-level 50/50 ablation | 0 | 0.5 | Entire set has three black queries half the time |
| NVS only | 1 | 0 | All query conditions black |
| Cleanup only | 0 | 0 | Stored noisy renders with pixel dropout |

Non-black-branch queries undergo online pixel dropout. `--min-drop 0 --max-drop
0.99` samples an integer percentage uniformly from 0 through 99, then independently
drops each pixel with that probability. `--max-drop 1` includes 100% dropout,
so even the cleanup branch can become fully black. Use 0.99 for the teacher's
explicit 50% per-query black-event recipe. Small renders or already-black inputs
can still become visually black without the explicit NVS switch. Diagnostic
dropout/task labels are **not** additional DiT inputs.

## Train or continue training

Example: continue a compatible 250k checkpoint for 400k additional updates, with
the final checkpoint numbered 650k. Supply a 1:1 HM3D/JHU manifest yourself:

```bash
python scripts/train.py \
  --data-root /data/omnicrafter \
  --train-manifest /data/omnicrafter/train_balanced.jsonl \
  --pose-manifest /data/omnicrafter/poses/manifest.json \
  --wan-root /weights/wan21 \
  --init-checkpoint /weights/step-250000.safetensors \
  --initial-step 250000 --steps 400000 --save-every 2000 \
  --gpus 0,1,2,3 --batch-size 1 --queries 3 \
  --height 256 --width 512 --nvs-ratio 0.5 --nvs-probability 0 \
  --min-drop 0 --max-drop 0.99 \
  --output /runs/omnicrafter_250k_to_650k
```

Add `--dry-run` to check required paths and print the underlying command without
loading WAN, allocating GPUs, or starting training. The launcher does not queue,
SSH, select free GPUs automatically, or terminate other processes. Select GPUs
that you are authorized to use. For persistent remote execution, run inside a
tmux session; the launcher itself is a foreground command.

The shipped config is the original-style **ZeRO-2 with GPU AdamW optimization**:
optimizer offload `none`, parameter offload `none`, bf16, CPU model initialization,
gradient checkpointing enabled, checkpointing offload disabled. CPU initialization
is not CPU optimization. Hardware capacity varies: the package does not guarantee
that every batch/query/resolution fits a particular GPU.

`--steps` in this wrapper means **additional** updates. The underlying trainer's
`--max_train_steps` is the cumulative endpoint. The output directory must be new,
to prevent accidental checkpoint overwrite. To resume after interruption, select
the saved checkpoint, set its step with `--initial-step`, and choose a new output
directory with the remaining additional steps. Training logs have timestamps;
loss and memory diagnostics are JSONL files.

Important: `.safetensors` continuation restores **LoRA and patch-embedding
weights only**. Optimizer moments, scheduler state, sampler state, and RNG state
are not restored; this is not a bit-exact optimizer resume. Loading the published
FrameCrafter initialization with initial step 0 is supported. This is not random
initialization of the 14B backbone.

The current trainer's `--batch-size` is a **logical batch per GPU**: it processes
sets sequentially on that GPU, scales/accumulates their losses, then updates once.
It is not a vectorized batch of eight panorama sets. With four GPUs and batch
size eight, one update consumes 32 sets and is expected to cost more than a
four-set update. The wrapper fixes gradient accumulation to 1 to match this path.

## Inference

Inference takes a JSON object with `pose_convention: "w2c_opencv"` and a `sets`
list. Each set has a safe unique `id`, exactly two `references`, and ordered
`targets`. Each reference has `image` and a 4×4 `w2c`; each target has `w2c`,
optional `image` (GT for scoring), and `noisy_image` (required only for cleanup).
Paths are relative to the **inference manifest's directory**, or absolute.
`toy_data/eval.json` generated above is a complete format example.

```bash
# Validate inputs without any weights or GPU/model loading:
python scripts/infer.py --manifest /data/eval.json --task cleanup --validate-only

# NVS: all query conditions are constructed as black images.
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --manifest /data/eval.json --checkpoint /weights/step-650000.safetensors \
  --wan-root /weights/wan21 --task nvs --height 256 --width 512 \
  --steps 20 --seed 20260904 --vram-limit 32 --output /results/nvs

# Cleanup: use the saved aligned renders; no online pixel dropout at evaluation.
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --manifest /data/eval.json --checkpoint /weights/step-650000.safetensors \
  --wan-root /weights/wan21 --task cleanup --height 256 --width 512 \
  --steps 20 --seed 20260904 --vram-limit 32 --output /results/cleanup
```

Each set is inferred jointly in one call; a 2ref3 set generates three images.
Query count comes from the manifest. The underlying implementation supports other
counts, but changing count changes context, ray normalization, memory, and possibly
quality; it is not automatically a fair comparison to the trained 2ref3 setting.
The target GT files are neither opened nor required during inference. Output:
`color/<set_id>/000000.png`, `000001.png`, `000002.png`, per-set metadata, and run
configuration/checksums. Existing output directories are rejected.

The standard teacher settings are 20 steps, CFG 1, empty prompt, scheduler shift
5, and bf16. Seeds use `base_seed + manifest_index`; keep the identical manifest
order for paired NVS/cleanup comparisons. This portable runner is not a claim of
bit-identical reproduction of every historic benchmark, whose original set-index
selection, model revision, environment, and metric versions must also match.

Inference VRAM management may offload frozen weights to CPU; `--vram-limit` is a
management budget, **not** a strict bound on total peak memory. Leave room for
activations. This inference offloading is separate from the no-offload training
optimizer config. Per-set timings exclude initial model loading, RGB/pose input
preparation, and PNG saving, but include the model call and its runtime transfers.
First-call warmup can affect timing; do not treat it as a standardized speed benchmark.

## Metrics

Install the optional metric dependencies, preferably in a separate environment
because DreamSim/PEFT dependencies can differ from the training environment:

```bash
python -m pip install -r requirements-metrics.txt
python scripts/score.py --manifest /data/eval.json --predictions /results/cleanup \
  --height 256 --width 512 --learned --output /results/cleanup_metrics.json
```

Without `--learned`, only PSNR, SSIM and WS-PSNR are computed. With it, VGG-LPIPS
and ensemble DreamSim cosine distance are added; metric weights may download on
first use. Higher PSNR/SSIM/WS-PSNR is better; lower LPIPS/DreamSim is better.
Metrics use complete RGB panoramas without validity masks, and arithmetic means
over target images. SSIM uses an 11-pixel Gaussian window with data range 1;
WS-PSNR weights ERP rows by cosine latitude. Prediction size must match exactly;
GT is Lanczos-resized if necessary. DreamSim uses its own pretrained preprocessing.
Versions are recorded in the report; match them before comparing close numbers
against historical evaluation results. Never silently skip failed predictions.

## Validation and provenance

`tests/test_package.py` exercises the actual CPU dataloader, streaming/eager
agreement, camera/ray layout, NVS/cleanup switches, non-mutation of source renders,
portable command construction, and a **mock-backbone** inference plumbing test.
It does not evaluate image quality or prove GPU memory fit. See
`provenance/validation.json` for checks actually run on this snapshot.

`provenance/source_manifest.json` records SHA-256 hashes of copied source files.
Historical server launchers are included as `.sh.txt` references only; they retain
the original machine-specific paths and are **not the portable entry points**.
`SHA256SUMS` covers the package files. The sibling ZIP is the convenient artifact
to send to a colleague.

This package includes source derived from [FrameCrafter](https://github.com/szqwu/FrameCrafter)
and DiffSynth-Studio. Their existing license is preserved in `code/LICENSE`
(Apache-2.0), with upstream README/code notices retained. This snapshot does not
grant new licenses to datasets or pretrained weights; obtain those separately
under their respective terms. No SCoPE camera-injection changes have been made.
