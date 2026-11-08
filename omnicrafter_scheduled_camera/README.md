# xliu253's concat + scheduled SCoPE-FF continuation (io104)

This directory is xliu253's trainer for the run
`mixed_concat_scheduled_scope_ff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io104`,
copied unchanged from `/cis/home/xliu253/Documents/PanoLagerNVS/omnicrafter_scheduled_camera_20261104/`.
Their own notes are in `README.txt`.

The same code also trains the UniRoPE v3 run on io105 (`CAMERA_VARIANT=unirope`). That run is on the branch
`omnicrafter-concat-scheduled-unirope`. Both branches share the snapshot commit, and differ only in `run_record/`, this README
and io105's launcher.

## Same as the running job

- All 133 files that the run's preflight hashed (`run_record/preflight.json`, `code_sha256`) match these copies byte for
  byte, checked on 2026-11-05. The 4 unhashed files are `code/LICENSE`, `code/README_FrameCrafter.md`,
  `code/pyproject.toml` and `provenance/RAYCRAFTER_TRAINING_NOTES.md`.
- `vendor/raycrafter_camera/{scope,unirope,rayrope}.py` are byte-identical to `src/raycrafter/cameras/` on main. The pin
  is RayCrafter 3458c26 (`provenance/source.json`).
- `code/` is the pinned FrameCrafter/DiffSynth snapshot; `bootstrap.py` puts it first on `sys.path`.
- `run_record/` holds the run's own records, copied from its output directory:
  - the launch command, preflight and protocol
  - the trainable parameters
  - the GPU correctness, gate-zero parity and optimizer-on-GPU checks
- `eval/` is xliu253's inference and evaluation of the step-100000 checkpoint on the shared benchmark, with its results.
  See `eval/README.md`.

## The run

| | |
|---|---|
| Output | `/cis/net/io81/data/xliu253/omniraycrafter/mixed_concat_scheduled_scope_ff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io104` |
| Launched | 2026-11-04 21:07 by `queue_scope_io104.sh` (`queue_training.py` → `preflight.py` → `run_job.py`) |
| Progress | step 89,020 on 2026-11-05 17:29, ~1.86 s/step |
| Steps | 50,000 → 350,000, `step-N.safetensors` every 2,000 |
| Init | their concat `mixed_projection_2ref3_..._io104/step-50000.safetensors` (sha256 `d62ecdfb…`); all 802 LoRA and 420-channel patch tensors restored strictly |
| Trainable | 1,282 tensors, 451.3M: LoRA 153.4M, patch embedding 8.6M, SCoPE-FF 289.4M (mostly the scale gate) |

**Launch flow.** The queue first runs `preflight.py` on the CPU. It then waits for io104's four cards to be idle: no compute
PIDs, ≤ 1 GiB used and ≤ 5% utilization, three checks in a row. Then `run_job.py` re-checks the hashes and runs
`gpu_smoke.py`, then `train.py`. Before the first update,
`verify_full_model.py` checks that the model with the gate at 0 is bitwise equal to the original concat model on all 4 ranks,
covering both projections and both tasks (`run_record/gate_zero_full_model_rank*.json`).

## What it trains

- **Camera schedule.** The concat input stays on. SCoPE-FF is added in all 40 self-attentions, after QK-norm and Wan's
  spatial RoPE: `Q = Q_rope + λ·scope_Q`, `K = K_rope + λ·scope_K`, V unchanged.
  - λ = clamp((step − 50000) / 10000, 0, 1). At λ = 0 the model runs the original `model_fn_wan_video`.
- **SCoPE-FF module.** `SightlineCoordinatePE(5120, mlp_hidden=64, dir_freqs=8, patch_size=True)`.
  - Inputs: 56 features per token (7 Plücker + 48 direction Fourier + patch angular size), through a 56→64→5120 MLP.
  - Initialization: output layer zero, hidden layer random.
  - Gate: `plucker_eps` 1e-3; no V residual.
  - The log-scale gate shift (p 0.3, range (−1.2, 1.6)) is active during training. The module is in train mode, and each
    block draws its own shift.
- **Cameras.**
  - Order ref1, ref2, q1, q2, q3, with no reference sorting. Everything is in the q3 camera frame.
  - SCoPE rays: translations divided by the scene's near depth. Rays are pinhole-K-only, so DL3DV lens distortion is
    ignored.
  - Concat rays are unchanged from the 50k model: mean camera distance = 1, OpenCV distortion inverted.
  - Temporal RoPE has zero phase.
- **Optimizer.** torch AdamW with a constant LR:
  - LoRA and patch embedding: lr 1e-5, wd 0.01.
  - SCoPE-FF: lr 1e-5, wd 0.01.
  - Gradient clipping 1.0.
- **Runtime.** DeepSpeed ZeRO-2, bf16, optimizer on GPU, no offload, 10M buckets. Gradient checkpointing on.
  1 set per GPU × 4 GPUs. Seed 20261023.
- **Objective.** Wan flow matching: the scheduler's 1000-step grid and loss weight, MSE on the 3 query latents only.
- **Restore.** `--restore-checkpoint` restores weights and the λ schedule. The optimizer state and RNG start fresh, and the
  data order starts over from the first set.

## Data

- **Manifest.** Their `mixed_rayrope_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/data_manifest/train.jsonl`, copied to
  `/dev/shm/xliu253_scheduled_scope_ff_20261104/` (sha256 `f8e68572…`) with its byte offsets.
  - 200k train sets: DL3DV 100k (perspective 288×512), HM3D 60k and JHU 40k (panorama 256×512). The 10 held-out JHU scenes
    are excluded.
  - Each row carries `near_depth` from Zhaoliang's tables (`/cis/home/zzhan288/RayCrafter/near_depth/`), and preflight
    checks every row against them.
- **Same rows as the concat run's manifest.** The same IDs in the same order, with identical poses and intrinsics. Only the
  image root and the `near_depth` fields differ.
- **Images.** PNGs under `/export/io104a/data/xliu253/omnicrafter_mixed_local_cache`, read through
  `image_path_map.shared.json`. On io104 that is the node's local NVMe.
- **Loading.** `dataset.py` reads each set inline on the training process (`num_workers=0`), which costs ~0.25 s per step:
  1. seek to the row and parse it;
  2. load the 2 reference and 3 ground-truth PNGs plus the 3 renders, with a PIL bilinear resize (no crop);
  3. build the CPU ray map.
- **Order.** `shuffle=True` with a generator seeded 20261023, sharded over 4 ranks by accelerate, so one pass is 50k steps.
  This is the concat run's order: continuation step s reads the set the concat run read at step s − 50000.
- **Tasks.** One draw per set:
  - NVS (p 0.5): all three queries black.
  - Cleanup: each render keeps a random 0–100% pixel dropout. All-black outcomes are redrawn.
  - A cleanup set with three empty renders is replaced by another set from the same scene, otherwise from the same dataset
    and resolution.
- **On the GPU, each step.** VAE encodes each frame separately (5 clean frames, then 2 refs + 3 query inputs for `y`), and
  T5 encodes the empty prompt. No CLIP.

## Running it from here

Not adapted: this is a record of the job as it runs. Paths and machines are hard-coded to xliu253's setup:
- `config.py`: `BASE` (output root, init checkpoint and source manifest), `LOCAL` (`/dev/shm` copy), `PYTHON` (their
  framecrafter env) and `GPU_UUIDS` (the queue refuses any other cards).
- `preflight.py`: the Wan root candidates in `/dev/shm`.
- `image_path_map.shared.json`, `queue_*.sh`, and the host lock in `/tmp`.

`queue_training.py` and `train.py` refuse an output that already has checkpoints. Pointed at xliu253's run, they would stop
there.
