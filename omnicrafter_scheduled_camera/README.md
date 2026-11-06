# Concat + scheduled SCoPE continuation on Skipjack

This is xliu253's io104 run, concat + scheduled SCoPE-FF (branch `omnicrafter-concat-scheduled-scope-ff`), with plain
SCoPE in place of SCoPE-FF. It is ported from their machines to Slurm on Skipjack and runs 50k → 100k. The same code
also trains the RayRoPE sibling (branch `omnicrafter-concat-scheduled-rayrope`, `CAMERA_VARIANT=rayrope`).

- xliu253's notes are in `README.txt`.
- The records of their run are in `run_record/`. They describe the SCoPE-FF run, not this one.

## Difference from xliu253's run

| | xliu253 (io104) | this run |
|---|---|---|
| camera module | SCoPE-FF: `SightlineCoordinatePE(5120, mlp_hidden=64, dir_freqs=8, patch_size=True)`, 56 inputs | plain SCoPE: `dir_freqs=0, patch_size=False`, the released 7 inputs (direction, normalized moment, log moment norm), as `configs/flux3_mix_scope.yaml` |
| rays | `plucker_tokens(..., patch_size=True)`, 7 channels | `patch_size=False`, 6 channels |
| steps | 50,000 → 350,000 | 50,000 → 100,000 (one pass over the 200k sets) |
| machine | io104, idle-GPU queue | Skipjack, 4×H100, `rchella4_videolincs`, one 32 h job |

Everything else is unchanged:
- **Init:** the concat step-50000 adapter, with all 802 tensors restored strictly.
- **Schedule:** λ ramps 0 → 1 over 10k steps. `Q = Q_rope + λ·SCoPE_Q`, `K = K_rope + λ·SCoPE_K`, V unchanged.
- **SCoPE settings:**
  - translations are divided by the scene's near depth, with q3 as the origin;
  - `plucker_eps` 1e-3, the log-scale gate shift, no V residual;
  - the zero-initialized output layer.
- **Optimizer:** LR 1e-5 and weight decay 0.01 for LoRA, the patch embedding and SCoPE. Constant LR, gradient clip 1.0.
- **Data:** the same manifest, order (seed 20261023) and per-set task draws, loaded inline.
- **Runtime:** ZeRO-2 bf16 with GPU AdamW.
- **Checks:** the gate-zero parity check on the 14B model.

Trainable: 802 inherited + 40 × 12 SCoPE = 1282 tensors.

## Porting

- **`config.py`:** Skipjack paths. `CAMERA_VARIANT` (`scope` or `rayrope`) must be set; it has no default.
- **Inputs:** copies of the concat step-50000 adapter and of `data_manifest/` are in `/weka/projects/rchella4/dli90/omnicrafter_outputs/`.
- **Images:** the manifest keeps the io104 image paths. `image_path_map.skipjack.json` maps them to the same files on weka:
  - DL3DV → `zzhan288/data/dl3dv_points/train`
  - HM3D and JHU → `zzhan288/data/framecrafter_5input`

  All 1,361,608 mapped paths exist. Their contents were not compared with io104's.
- **Near depth:** comes from `data_manifest/near_depth/`, whose hashes match `summary.json`.
- **Launcher:** `preflight.py` reads the manifest in place. `train_4gpu.sbatch` replaces `queue_training.py`, `run_job.py` and the `queue_*.sh` scripts. It runs the preflight, then `gpu_smoke.py` on a fresh start, then `train.py`.
- **Full-state resume (`checkpoints.py`, `data_position.py`):** xliu253's `--restore-checkpoint` resets AdamW, the RNG and the data order. This run instead keeps `resume/` every 2000 steps and 30 min before the walltime. It holds:
  - the DeepSpeed weights and AdamW;
  - the scheduler and every RNG;
  - each rank's place in the shuffled data.

  A resubmitted job continues with the same sets and task draws, and the checksum is verified on load. On a fresh start, the trainer saves and reloads this state once at step 50010, which checks the resume path early.

## Run

Use the worktree at `/weka/home/jhu/dli90/RayCrafter_runs/concat-scheduled-scope`. Jobs read the code when they start, so they must not see a branch switch.

```bash
cd /weka/home/jhu/dli90/RayCrafter_runs/concat-scheduled-scope
CAMERA_VARIANT=scope sbatch -J oc-sched-scope omnicrafter_scheduled_camera/train_4gpu.sbatch
```

Resubmit the same command to continue after a crash or a walltime stop. A shorter `-t` is fine for the remainder; set `TIME_LIMIT_HOURS` to 30 min less. Once the run reaches 100k, a resubmit only reports completion.

CPU tests:

```bash
cd omnicrafter_scheduled_camera
CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 CAMERA_VARIANT=scope \
  /weka/projects/rchella4/dli90/envs/omnicrafter/bin/python -B -m unittest test_scheduled test_dataset_resume test_data_position -v
```

## Outputs

Everything goes in `/weka/projects/rchella4/dli90/omnicrafter_outputs/mixed_concat_scheduled_scope_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus50k_skipjack/`:

- **Checkpoints:** `step-N.safetensors` every 2000 steps, holding the trainable tensors with the camera contract and schedule in the metadata.
- **`resume/`:** the full state.
- **Logs:** `train_loss.jsonl` every 10 steps (loss, λ, task and dataset fractions, rank 0's set id), `progress.json`, and `logs/`.
- **Records:** `preflight.json`, `protocol_*.json`, `trainable_parameters.json`.
- **Checks:** `gpu_correctness_rank*.json`, `gate_zero_full_model_rank*.json`, `gpu_optimizer_rank*.json`.

The Slurm log is `bash_outputs/oc-sched-scope-<job>.out` in the main checkout.
