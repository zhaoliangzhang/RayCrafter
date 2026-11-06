# Concat + SCoPE-FF continuation without a schedule, on Skipjack

This is xliu253's io104 run, concat + scheduled SCoPE-FF (branch `omnicrafter-concat-scheduled-scope-ff`), without
the injection schedule: λ is 1 from the first update. It uses the same Skipjack port as the two scheduled siblings:
- branch `omnicrafter-concat-scheduled-scope`, plain SCoPE;
- branch `omnicrafter-concat-scheduled-rayrope`, RayRoPE.

All three run 50k → 100k from the same concat checkpoint.

- xliu253's notes are in `README.txt`.
- The records of their run are in `run_record/`. They describe the scheduled SCoPE-FF run, not this one.

## Difference from xliu253's run

| | xliu253 (io104) | this run |
|---|---|---|
| injection | λ = clamp((step − 50000)/10000, 0, 1): 1e-4 at the first update, 1 at 60k | no schedule: λ = 1 from the first update (`RAMP_STEPS = 0`) |
| steps | 50,000 → 350,000 | 50,000 → 100,000 (one pass over the 200k sets) |
| machine | io104, idle-GPU queue | Skipjack, 4×H100, `rchella4_videolincs`, one 32 h job |

The model before the first update is still the concat model. SCoPE-FF's output layer starts at zero, so the camera term
added to Q and K is exactly zero until training moves it. The ramp only scaled that term, and its gradients, by λ for
the first 10k steps.

Everything else is unchanged from xliu253's run:
- **Init:** the concat step-50000 adapter, with all 802 tensors restored strictly.
- **Injection:** `Q = Q_rope + SCoPE_Q`, `K = K_rope + SCoPE_K`, V unchanged. Wan's spatial RoPE is kept, with zero temporal phase.
- **SCoPE-FF:**
  - `SightlineCoordinatePE(5120, mlp_hidden=64, dir_freqs=8, patch_size=True)`, with 56 MLP inputs: the released 7, plus 8 direction frequencies and the patch's angular size;
  - 7-channel rays from `plucker_tokens(..., patch_size=True)`;
  - translations are divided by the scene's near depth, with q3 as the origin;
  - `plucker_eps` 1e-3, the log-scale gate shift, no V residual, the zero-initialized output layer.
- **Optimizer:** LR 1e-5 and weight decay 0.01 for LoRA, the patch embedding and SCoPE-FF. Constant LR, gradient clip 1.0.
- **Data:** the same manifest, order (seed 20261023) and per-set task draws, loaded inline.
- **Runtime:** ZeRO-2 bf16 with GPU AdamW.
- **Checks:** the gate-zero parity check on the 14B model.

Trainable: 802 inherited + 40 × 12 SCoPE-FF = 1282 tensors.

## Changes from the scheduled SCoPE sibling

- **`config.py`:** `CAMERA_VARIANT` must be `scope_ff`. `RAMP_STEPS = 0`, and `injection_weight` gives 1 for every update when the ramp is 0. The output directory is `mixed_concat_scopeff_…`.
- **Camera modules:** SCoPE-FF settings in `camera.py` and `wan_camera.py`.
- **Contract:** records SCoPE-FF, `camera_injection: concat_scope_ff` and `injection_schedule: none_lambda_1_from_first_update`. Adapters record a `constant` schedule.
- **New check (`verify_full_model.py`):** on the 14B model before the first update, the λ = 1 camera path with the fresh SCoPE-FF must equal the concat model. The check expects bitwise equality and fails above 1% of the output's max. It reports `lambda_one_zero_init_*` in `gate_zero_full_model_rank*.json`.
- **New CPU test (`test_scheduled.py`):** the same check on a tiny model, requiring bitwise equality.

## Porting

As in the scheduled siblings:
- **Inputs:** copies of the concat step-50000 adapter and of `data_manifest/` are in `/weka/projects/rchella4/dli90/omnicrafter_outputs/`.
- **Images:** `image_path_map.skipjack.json` maps the manifest's io104 image paths to the same files on weka:
  - DL3DV → `zzhan288/data/dl3dv_points/train`
  - HM3D and JHU → `zzhan288/data/framecrafter_5input`
- **Near depth:** from `data_manifest/near_depth/`, whose hashes match `summary.json`.
- **Launcher:** `train_4gpu.sbatch` runs `preflight.py`, then `gpu_smoke.py` on a fresh start, then `train.py`.
- **Full-state resume:** `resume/` every 2000 steps and 30 min before the walltime. It holds:
  - the DeepSpeed weights and AdamW;
  - the scheduler and every RNG;
  - each rank's place in the shuffled data.

  The checksum is verified on load. On a fresh start, the trainer saves and reloads this state once at step 50010.

## Run

Use the worktree at `/weka/home/jhu/dli90/RayCrafter_runs/concat-scopeff`. Jobs read the code when they start, so they
must not see a branch switch.

```bash
cd /weka/home/jhu/dli90/RayCrafter_runs/concat-scopeff
CAMERA_VARIANT=scope_ff sbatch -J oc-concat-scopeff omnicrafter_scheduled_camera/train_4gpu.sbatch
```

Resubmit the same command to continue after a crash or a walltime stop. A shorter `-t` is fine for the remainder; set
`TIME_LIMIT_HOURS` to 30 min less. Once the run reaches 100k, a resubmit only reports completion.

CPU tests:

```bash
cd omnicrafter_scheduled_camera
CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 CAMERA_VARIANT=scope_ff \
  /weka/projects/rchella4/dli90/envs/omnicrafter/bin/python -B -m unittest test_scheduled test_dataset_resume test_data_position -v
```

## Outputs

Everything goes in `/weka/projects/rchella4/dli90/omnicrafter_outputs/mixed_concat_scopeff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus50k_skipjack/`:

- **Checkpoints:** `step-N.safetensors` every 2000 steps, holding the trainable tensors with the camera contract and schedule in the metadata.
- **`resume/`:** the full state.
- **Logs:** `train_loss.jsonl` every 10 steps (loss, λ, task and dataset fractions, rank 0's set id), `progress.json`, and `logs/`.
- **Records:** `preflight.json`, `protocol_*.json`, `trainable_parameters.json`.
- **Checks:** `gpu_correctness_rank*.json`, `gate_zero_full_model_rank*.json`, `gpu_optimizer_rank*.json`.

The Slurm log is `bash_outputs/oc-concat-scopeff-<job>.out` in the main checkout.
