# Transfer, configure and run on another cluster

Nothing below has been submitted to a new cluster. Replace `YOUR_USER`, hostnames
and paths with your real allocation. Confirm scheduler, partition/account, CUDA
driver, GPU memory, system RAM and storage quota first. The examples assume one
node with four allocated GPUs for training. They do not need io104/io95 hostnames.
The destination has now been identified as **Skipjack/ARCH Slurm**: read
[SKIPJACK.md](SKIPJACK.md) for the real PI-based storage layout and batch templates.

## 1. Transfer the package and verify it

Copy the updated `OmniCrafter_cluster_handoff_20261028/` directory into a working
directory. The original same-named October 28 `.tar.gz` is retained unchanged
and does **not** include the October 29 portable SCoPE training fix.
From its root:

```bash
python tools/package_inventory.py
python -B -m unittest discover -s tests -v
python tools/audit_manifests.py
```

Do this before editing packaged files. Keep an untouched copy of the archive and
store edited configurations under new filenames. Later inventory differences
are expected for intentional edits; do not hide them by claiming the original hash.

## 2. Copy current mixed training images

The consolidated cache is already verified and avoids fetching images from many
old source servers. Its RGB payload is **747.2 GB decimal / 695.9 GiB**, excluding
additional filesystem overhead and new run/checkpoint storage. Allow extra space
for weights, evaluations and hundreds of checkpoint files.

Example to execute from a destination login/transfer node that can reach io104:

```bash
mkdir -p /scratch/YOUR_USER/omnicrafter/train_cache/images
rsync -aL --partial --info=progress2 \
  io104:/export/io104a/data/xliu253/omnicrafter_mixed_local_cache/images/ \
  /scratch/YOUR_USER/omnicrafter/train_cache/images/
```

If direct SSH is not possible, use your cluster's approved transfer service or
staging endpoint. Preserve the directory tree; do not flatten image basenames.
`-L` materializes symlink targets. **No `--delete` is needed.** Run a checksum dry
run afterward (`rsync -acnL --itemize-changes ...`) and require no differences.

Edit a copy of `configs/train_paths.example.json` to map the single old local
cache prefix to the new cache prefix. Relocate the included manifest and rebuild
its byte-offset index:

```bash
python tools/relocate_manifest.py \
  --kind train --input metadata/train_cached.jsonl.gz \
  --mapping configs/train_paths.local.json \
  --output /scratch/YOUR_USER/omnicrafter/train_manifest/train.jsonl \
  --check-files
```

This writes `train.jsonl`, `train.offsets.npy` and `summary.json`. It refuses an
existing output manifest. Only runtime RGB paths change; IDs, sample order,
repeated entries, split, camera matrices/calibration and image dimensions stay
the same. Training provenance fields such as `source_manifest` and
`prior_metadata` retain their original addresses; the trainer does not open them.

The mixed manifest embeds poses, so this recipe does not need the legacy pose
NPZ tree at runtime. The **older panorama-only** loader does need that pose tree.
Never reuse an offsets file after rewriting a JSONL: even a one-character path
change invalidates downstream byte positions.

## 3. Copy the exact evaluation assets

Generate a deduplicated list **on a machine with access to the original paths**:

```bash
python tools/plan_eval_transfer.py --output /tmp/omnicrafter_eval.files0
```

This includes reference RGBs, original GT, original cached metric GT, noisy priors
and prior metadata. It does not include predictions or WAN. Example transfer
from that source machine, preserving the selected absolute directory structure
below a new `eval_assets` root:

```bash
rsync -aL --partial --from0 --files-from=/tmp/omnicrafter_eval.files0 \
  / NEW_CLUSTER:/scratch/YOUR_USER/omnicrafter/eval_assets/
```

Only paths in the generated list are transferred. Do not replace this with an
unfiltered copy of `/`. If outbound SSH is unavailable, stage the same selected
tree using your transfer service. Internal links inside copied metadata JSON
remain historical provenance; the evaluation reads the new top-level manifest.

Edit `configs/eval_paths.example.json`; the normal mapping is old `/cis` to
`/scratch/YOUR_USER/omnicrafter/eval_assets/cis`. Then:

```bash
python tools/verify_assets.py --mapping configs/eval_paths.local.json
python tools/relocate_manifest.py \
  --kind eval --input metadata/eval_manifest.json \
  --mapping configs/eval_paths.local.json \
  --output /scratch/YOUR_USER/omnicrafter/eval_manifest/manifest.json \
  --check-files
```

The checksum verifier checks every saved original RGB input hash. Do not resize
the images during transfer. Absolute symlinks alone are not transferable data.

## 4. Stage model and metric weights

Copy a **trusted** complete WAN directory; the portable wrappers expect:

```text
weights/wan21/Wan-AI/
  Wan2.1-I2V-14B-480P/
    diffusion_pytorch_model-00001-of-00007.safetensors
    ... all seven DiT shards ...
    models_t5_umt5-xxl-enc-bf16.pth
    Wan2.1_VAE.pth
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
  Wan2.1-T2V-1.3B/google/umt5-xxl/
    tokenizer_config.json
    ... all tokenizer files ...
```

`wan_root` points to `weights/wan21`, not the I2V subdirectory. Copy the published
`framecrafter.safetensors` adapter separately for training initialization, and
the chosen trained adapter for continuation/evaluation. Published adapter SHA:

`fa22ffdb9a54d5eb029618bcd424af1623ee6d64eb6277db7f7c478ab5b92524`

Concat adapter files are about310MiB; SCoPE checkpoints are about801MiB because
they also contain the camera modules. Neither is a full14B checkpoint. Preserve SHA256 for
every transferred weight. Original `.pth`/`.pt` files can use pickle-based loaders;
load only trusted files. Do not change formats during an exact migration.

For metrics copy the DreamSim ensemble cache, Clean-FID Inception checkpoint,
and the torch cache containing LPIPS/VGG weights. For PanoVGGT regeneration also
copy its full checkpoint and initialization caches; see `panovggt/README.md`.
The portable evaluation workers set HF/Transformers offline flags. Perform
dependency/cache preflight on a permitted networked node before offline GPU jobs.

## 5. Environments

Use separate **generation/training**, **metrics**, and optionally **PanoVGGT**
environments. Their torch/transformers/peft versions differ. Blindly upgrading
everything into one environment risks changing results or breaking imports.

The observed training versions are recorded in
`provenance/training_environment_observed.json`; the older matching constraints
are in `panorama/requirements-observed.txt`. They describe the source machine,
not a guarantee that wheels/drivers fit your destination. Choose a compatible
CUDA torch/torchvision pair, then install the bundled source rather than an
arbitrary upstream DiffSynth revision. Example after activating that environment:

```bash
python -m pip install -c panorama/requirements-observed.txt -e snapshots/mixed/code
python -m pip install PyYAML
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -B snapshots/mixed/test_protocol.py
```

DeepSpeed needs an appropriate CUDA/compiler toolchain. CPU-only schema tests
do not prove a14B model fits GPU or host RAM. The source run used four96GB-class
cards; a different GPU size requires a measured smoke test, not an assumption.

For metrics use the version table in [EVALUATION.md](EVALUATION.md), including
LPIPS-VGG and the official DreamSim ensemble. A source environment that already
works may be exported using your cluster's supported container/environment
workflow, but do not copy a conda directory blindly and assume absolute prefixes
or CUDA extensions remain valid. This handoff does not create a container image.

## 6. Launch current mixed training

Copy/edit `configs/train_mixed.example.json`. All filesystem values should be
absolute destination-cluster paths. The fixed recipe is fourGPUs, batch1/GPU,
sample-level50% NVS,2ref3, ERP256×512 H×W and perspective288×512 H×W.
The example selects `architecture: "concat"`; older configs without that key
also retain concat. Select SCoPE explicitly as described below.

```bash
python tools/train_mixed.py --config configs/train_mixed.local.json --dry-run
python tools/train_mixed.py --config configs/train_mixed.local.json
```

For a **weights-only continuation** from50k to300k, set:

```json
{
  "restore_checkpoint": "/scratch/YOUR_USER/omnicrafter/weights/step-50000.safetensors",
  "initial_step": 50000,
  "final_step": 300000,
  "save_every": 2000
}
```

Keep the other configuration fields. Use a **new output directory**. This performs
250k additional optimizer updates, with reset optimizer/scheduler/RNG state.
For an initial GPU smoke test use a new scratch output, `initial_step=0`,
`final_step=100`, `save_every=100`; inspect that all projection/task cases execute,
loss is finite, and optimizer states are on CUDA. Do not count this smoke run as
a production continuation or overwrite its checkpoints.

The supplied Slurm template is deliberately incomplete: fill account/partition,
walltime, memory request and environment activation. Export `PACKAGE_ROOT` and
`TRAIN_CONFIG` with absolute paths, then submit using your scheduler. Do not add
an8-process `srun` around an already four-process Accelerate launch. Use one
scheduler task on one node, letting Accelerate spawn its four ranks.

On a direct-SSH cluster with authorized exclusive GPUs, use a tmux session and
set `CUDA_VISIBLE_DEVICES` explicitly. On a scheduled cluster **retain the
scheduler's CUDA_VISIBLE_DEVICES**; the wrapper respects it. It does not select
free GPUs, log in to old hosts, submit jobs, or enable memory parking.

### SCoPE training

Copy `configs/train_scope.example.json` to `configs/train_scope.local.json` and
edit all paths for your cluster. Use the **same relocated mixed manifest**, WAN
root and exact published FrameCrafter adapter as above. Keep:

```json
{
  "architecture": "scope",
  "restore_checkpoint": null,
  "initial_step": 0,
  "final_step": 300000,
  "save_every": 2000,
  "num_gpus": 4,
  "batch_per_gpu": 1
}
```

This snippet is not a complete config; retain the path fields from the example.
`base_initialization` must still point to the original **420-channel published**
adapter. Do not manually replace it with a trained SCoPE checkpoint or slice it
in place. On a real launch, the wrapper converts it on CPU once, writes
`OUTPUT/initialization/framecrafter_image36.safetensors`, records the effective
config in `OUTPUT/runtime/train_config.json`, then starts four Accelerate ranks
with `--architecture scope`. Dry-run validates but does not convert or allocate
CUDA memory.

```bash
python tools/train_mixed.py --config configs/train_scope.local.json --dry-run
# Run inside an authorized four-GPU allocation:
python tools/train_mixed.py --config configs/train_scope.local.json
```

For Skipjack, use the existing batch template with the SCoPE JSON:

```bash
export PACKAGE_ROOT=/absolute/path/to/OmniCrafter_cluster_handoff_20261028
export TRAIN_CONFIG="$PACKAGE_ROOT/configs/train_scope.local.json"
export TRAIN_PYTHON=/absolute/path/to/your/environment/bin/python
# Edit account/partition/memory/time in the template before submitting.
sbatch "$PACKAGE_ROOT/configs/skipjack_train.sbatch"
```

The data mixture remains 100k DL3DV / 60k HM3D / 40k JHU; image resolutions and
sample-level 50% all-black NVS / 50% noisy-render cleanup are unchanged. ZeRO2,
GPU AdamW, CPU initialization and gradient checkpointing remain unchanged. No
optimizer or activation CPU offload and no GPU memory-parking are enabled.

For continuation, set `restore_checkpoint` to a trained **SCoPE** checkpoint and
`initial_step` to its recorded step, with a new output directory. The loader
requires its 36-channel patch plus all 400 camera parameters (1,202 tensors
total), and restores their values. This remains a weights-only continuation,
not an exact optimizer-state resume. A 420-channel concat checkpoint cannot be
used as `restore_checkpoint` for SCoPE.

Start with a new-output 100-step / save-every-100 smoke allocation, as above.
Confirm `camera_injection=scope`, SCoPE parameter updates, both projections/tasks
and CUDA optimizer states. No full distributed GPU smoke on Skipjack is claimed
by the portable package. **`tools/evaluate.py` still does not support SCoPE**;
it will reject that architecture/checkpoint rather than run the wrong model.

## 7. Run or resume the matched mixed evaluation

Edit `configs/eval_mixed.example.json`, including `metric_python` and the number
of allocated GPU workers. Then:

```bash
python tools/evaluate.py --config configs/eval_mixed.local.json --dry-run
python tools/evaluate.py --config configs/eval_mixed.local.json --phase prepare
python tools/evaluate.py --config configs/eval_mixed.local.json --phase smoke
python tools/evaluate.py --config configs/eval_mixed.local.json --phase all
```

`prepare` checks file availability and GT dimensions and creates reference/GT/
prior links; it does not load WAN. `all` performs sharded inference, metrics and
final aggregation. You can run `--phase inference`, `--phase metrics`, or
`--phase aggregate` separately. Resume with the **same** config and output; completed
set-task predictions and metric shards are reused. Do not change GPU shard count
halfway through a metric run. Logs have timestamps under `output/logs/`.

This wrapper currently reproduces the **full mixed benchmark**, not arbitrary
dataset subsets; it fails closed on a changed count or changed seed/order.
The new-cluster GPU execution and numeric metric parity remain to be verified.
`smoke` uses one allocated GPU and the first set of each dataset (four sets,
both tasks,24 predictions), then stops; these outputs are retained by the full
run. It does not score metrics. Start with this smoke before the full benchmark.

## 8. Other experiments / what to ask next

For a panorama-only historical recipe, read `panorama/README.md` and supply its
legacy per-target JSONL plus pose manifest/NPZs. Its train wrapper uses
**additional** `--steps`, unlike the mixed wrapper's cumulative `final_step`.
Its generic scoring wrapper is not a substitute for the exact mixed benchmark.

When asking for a new command, provide:

```text
Experiment: concatenated-ray OmniCrafter / SCoPE / original FrameCrafter / distilled
Initialization: exact checkpoint path and SHA, or published initialization
Data identity: exact train and eval manifest hashes, not just "HM3D/JHU"
Task switch: per-query or per-sample; probability; cleanup dropout bounds
Views + resolution: refs/queries, H×W per projection
Compute: scheduler/partition/account, GPU count/type/memory, RAM, walltime
Storage: data/weights local cache, output root, scratch persistence policy
Updates: initial step, cumulative endpoint, save interval, batch/GPU, accumulation
Evaluation: target subset, seed, denoising settings, metric variants
```

The useful combination is **this package as the source of truth plus small
versioned experiment configurations**, with new commands generated from those
configurations when needed. Chat history alone should not be the experiment log.
