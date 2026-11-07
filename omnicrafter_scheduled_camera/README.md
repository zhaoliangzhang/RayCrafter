# Concat + scheduled RayRoPE continuation on Skipjack

This is xliu253's io105 run, concat + scheduled UniRoPE v3 (branch `omnicrafter-concat-scheduled-unirope`), with
RayRoPE in place of UniRoPE. It is ported from their machines to Slurm on Skipjack and runs 50k → 100k. The same code
also trains the SCoPE sibling (branch `omnicrafter-concat-scheduled-scope`, `CAMERA_VARIANT=scope`).

- xliu253's notes are in `README.txt`.
- The records of their run are in `run_record/`, and its launcher is in `io105_launch/`. They describe the UniRoPE run, not this one.

## Difference from xliu253's run

| | xliu253 (io105) | this run |
|---|---|---|
| camera module | UniRoPE v3: `prepare_unirope(footprint=True, adaptive_freqs=True)` + `unirope_attention` | RayRoPE: `prepare_geometry` + `rayrope_attention` from the same pinned `vendor/raycrafter_camera/rayrope.py`, as `configs/flux3_mix_rayrope.yaml` |
| depth head attribute | `self_attn.unirope_depth` | `self_attn.rayrope_depth` (same `DepthHead`) |
| steps | 50,000 → 350,000 | 50,000 → 100,000 (one pass over the 200k sets) |
| machine | io105, shared-admission launcher | Skipjack, 4×H100, `rchella4_videolincs`, one 36 h job |
| `torch.compile` | on | off: RayRoPE's compiled functions run eagerly (see Porting) |

How RayRoPE encodes each token:
- The query camera's center, plus three patch-corner rays at the predicted depth.
- Each ray contributes its direction xy and its depth.
- 5 frequencies rotate 120 of each head's 128 channels.
- V is rotated like K, and the output is rotated back (`apply_vo`).
- Attention runs once per query camera.
- Panorama batches use signed z/w for depth, because half of a panorama is behind any camera. Pinhole batches use the reference |z|/w.

Everything else is unchanged:
- **Init:** the concat step-50000 adapter, with all 802 tensors restored strictly.
- **Schedule:** λ ramps 0 → 1 over 10k steps, as `(1−λ)·(Wan spatial-RoPE attention) + λ·(RayRoPE attention)` before the shared output projection.
- **Poses:**
  - the ref1–ref2 distance is 1;
  - the world frame moved to q3, which doesn't change RayRoPE's per-camera coordinates;
  - no reference sorting.
- **Depth head:** `DepthHead` per block with log depth 0 and σ 3. LR 4e-4, weight decay 0.5.
- **Optimizer:** LR 1e-5 and weight decay 0.01 for LoRA and the patch embedding. Constant LR, gradient clip 1.0.
- **Data:** the same manifest, order (seed 20261023) and per-set task draws, loaded inline.
- **Runtime:** ZeRO-2 bf16 with GPU AdamW.
- **Checks:** the gate-zero parity check on the 14B model.

Trainable: 802 inherited + 40 × 2 depth-head tensors = 882.

## Porting

- **`config.py`:** Skipjack paths. `CAMERA_VARIANT` (`scope` or `rayrope`) must be set; it has no default.
- **Inputs:** copies of the concat step-50000 adapter and of `data_manifest/` are in `/weka/projects/rchella4/dli90/omnicrafter_outputs/`.
- **Images:** the manifest keeps the io104 image paths. `image_path_map.skipjack.json` maps them to the same files on weka:
  - DL3DV → `zzhan288/data/dl3dv_points/train`
  - HM3D and JHU → `zzhan288/data/framecrafter_5input`

  All 1,361,608 mapped paths exist. Their contents were not compared with io104's.
- **Near depth:** comes from `data_manifest/near_depth/`, whose hashes match `summary.json`. RayRoPE doesn't use it, but preflight still checks it.
- **Launcher:** `preflight.py` reads the manifest in place. `train_4gpu.sbatch` replaces `queue_training.py`, `run_job.py` and the `queue_*.sh` scripts. It runs the preflight, then `gpu_smoke.py` on a fresh start, then `train.py`.
- **No `torch.compile`:** `train_4gpu.sbatch` sets `TORCH_COMPILE_DISABLE=1`, as `omnicrafter_unirope` does. Triton builds
  a C stub against `Python.h`, which this env's system Python doesn't have on Skipjack. The first job, 1597642, failed in
  `gpu_smoke.py` on it. The functions are the same, but eager runs at about 2.2 s/step, so the walltime is 36 h.
- **Full-state resume (`checkpoints.py`, `data_position.py`):** xliu253's `--restore-checkpoint` resets AdamW, the RNG and the data order. This run instead keeps `resume/` every 2000 steps and 30 min before the walltime. It holds:
  - the DeepSpeed weights and AdamW;
  - the scheduler and every RNG;
  - each rank's place in the shuffled data.

  A resubmitted job continues with the same sets and task draws, and the checksum is verified on load. On a fresh start, the trainer saves and reloads this state once at step 50010, which checks the resume path early.

## Run

Use the worktree at `/weka/home/jhu/dli90/RayCrafter_runs/concat-scheduled-rayrope`. Jobs read the code when they start, so they must not see a branch switch.

```bash
cd /weka/home/jhu/dli90/RayCrafter_runs/concat-scheduled-rayrope
CAMERA_VARIANT=rayrope sbatch -J oc-sched-rayrope omnicrafter_scheduled_camera/train_4gpu.sbatch
```

Resubmit the same command to continue after a crash or a walltime stop. A shorter `-t` is fine for the remainder; set `TIME_LIMIT_HOURS` to 30 min less. Once the run reaches 100k, a resubmit only reports completion.

CPU tests:

```bash
cd omnicrafter_scheduled_camera
CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 CAMERA_VARIANT=rayrope \
  /weka/projects/rchella4/dli90/envs/omnicrafter/bin/python -B -m unittest test_scheduled test_dataset_resume test_data_position -v
```

## Outputs

Everything goes in `/weka/projects/rchella4/dli90/omnicrafter_outputs/mixed_concat_scheduled_rayrope_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus50k_skipjack/`:

- **Checkpoints:** `step-N.safetensors` every 2000 steps, holding the trainable tensors with the camera contract and schedule in the metadata.
- **`resume/`:** the full state.
- **Logs:** `train_loss.jsonl` every 10 steps (loss, λ, task and dataset fractions, rank 0's set id, `depth/log_depth_bias` and `depth/sigma_bias`), `progress.json`, and `logs/`.
- **Records:** `preflight.json`, `protocol_*.json`, `trainable_parameters.json`.
- **Checks:** `gpu_correctness_rank*.json`, `gate_zero_full_model_rank*.json`, `gpu_optimizer_rank*.json`.

The Slurm log is `bash_outputs/oc-sched-rayrope-<job>.out` in the main checkout.
