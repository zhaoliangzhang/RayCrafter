# xliu253's concat + scheduled UniRoPE v3 continuation (io105)

This directory is xliu253's trainer for the run
`mixed_concat_scheduled_unirope_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io105`,
copied unchanged from `/cis/home/xliu253/Documents/PanoLagerNVS/omnicrafter_scheduled_camera_20261104/`.
Their own notes are in `README.txt`. The trainer runs this variant with `CAMERA_VARIANT=unirope`. `config.py` defaults to
`scope_ff`, so the variable must be set.

The same code also trains the SCoPE-FF run on io104, which is on the branch `omnicrafter-concat-scheduled-scope-ff`. Both
branches share the snapshot commit, and differ only in `run_record/`, this README and `io105_launch/`.

## Same as the running job

- All 133 files that the run's preflight hashed (`run_record/preflight.json`, `code_sha256`) match these copies byte for
  byte, checked on 2026-11-05. The 4 unhashed files are `code/LICENSE`, `code/README_FrameCrafter.md`,
  `code/pyproject.toml` and `provenance/RAYCRAFTER_TRAINING_NOTES.md`.
- `io105_launch/launch_now.py` is the launcher this run actually used, unchanged. Its sha256 `b65247ac…` matches
  `run_record/launch.json`.
- `vendor/raycrafter_camera/{scope,unirope,rayrope}.py` are byte-identical to `src/raycrafter/cameras/` on main. The pin
  is RayCrafter 3458c26 (`provenance/source.json`).
- `code/` is the pinned FrameCrafter/DiffSynth snapshot; `bootstrap.py` puts it first on `sys.path`.
- `run_record/` holds the run's own records, copied from its output directory:
  - the launch command, including the shared-admission snapshot, preflight and protocol
  - the trainable parameters
  - the GPU correctness, gate-zero parity and optimizer-on-GPU checks

## The run

| | |
|---|---|
| Output | `/cis/net/io81/data/xliu253/omniraycrafter/mixed_concat_scheduled_unirope_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io105` |
| Launched | 2026-11-04 21:10 by `io105_launch/launch_now.py`, not the idle-GPU queue |
| Progress | step 94,320 on 2026-11-05 17:29, ~1.59 s/step |
| Steps | 50,000 → 350,000, `step-N.safetensors` every 2,000 |
| Init | their concat `mixed_projection_2ref3_..._io104/step-50000.safetensors` (sha256 `d62ecdfb…`); all 802 LoRA and 420-channel patch tensors restored strictly |
| Trainable | 882 tensors, 162.4M: LoRA 153.4M, patch embedding 8.6M, depth heads 0.4M |

**Launch flow.** `launch_now.py` admitted the job beside xliu253's running concat + SCoPE-FF + UniRoPE v3 job: PIDs
663291–663294, ~53 GB per GPU, which stopped at step 67,220 at 21:14.
- It allows only those four PIDs and requires ≥ 64 GiB free on every card.
- It re-checks the preflight hashes, then runs `gpu_smoke.py`, then `train.py`.

Before the first update, `verify_full_model.py` checks that the model with the gate at 0 is bitwise equal to the original
concat model on all 4 ranks, covering both projections and both tasks (`run_record/gate_zero_full_model_rank*.json`).

## What it trains

- **Camera schedule.** The concat input stays on. In all 40 self-attentions, after QK-norm, the attention output is
  `(1−λ)·(Wan spatial-RoPE attention) + λ·(UniRoPE attention)`, taken before the shared output projection.
  - λ = clamp((step − 50000) / 10000, 0, 1). At λ = 0 the model runs the original `model_fn_wan_video`.
  - At λ = 1 only UniRoPE runs, so Wan's spatial RoPE is no longer used.
- **UniRoPE v3.** `unirope_attention` with `footprint`, `adaptive_freqs` and `apply_vo` on. It rotates 120 of each head's
  128 channels; the other 8 pass through. With `apply_vo`, V is rotated like K and the output is rotated back, as in RayRoPE.
  - Each token is its patch's camera center plus 3 corner points at the predicted depth.
  - Direction frequencies adapt to each query camera's angular patch size.
  - Attention runs once per query camera, in that camera's frame.
  - A `DepthHead` per block gives log-depth 0 and σ 3 at the start (depth = 1 reference baseline), clamped to 0.01–100.
- **Cameras.**
  - Order ref1, ref2, q1, q2, q3, with no reference sorting.
  - UniRoPE pose scale: the ref1–ref2 distance is 1, and the world frame is moved to q3. The origin doesn't change the
    scores, because attention is computed per query camera.
  - UniRoPE rays are pinhole-K-only, so DL3DV lens distortion is ignored.
  - Concat rays are unchanged from the 50k model: q3 frame, mean camera distance = 1, OpenCV distortion inverted.
  - Temporal RoPE has zero phase.
- **Optimizer.** torch AdamW with a constant LR:
  - LoRA and patch embedding: lr 1e-5, wd 0.01.
  - Depth heads: lr 4e-4, wd 0.5.
  - Gradient clipping 1.0.
- **Runtime.** DeepSpeed ZeRO-2, bf16, optimizer on GPU, no offload, 10M buckets. Gradient checkpointing on.
  1 set per GPU × 4 GPUs. Seed 20261023.
- **Objective.** Wan flow matching: the scheduler's 1000-step grid and loss weight, MSE on the 3 query latents only.
- **Restore.** `--restore-checkpoint` restores weights and the λ schedule. The optimizer state and RNG start fresh, and the
  data order starts over from the first set.

## Data

- **Manifest.** Their `mixed_rayrope_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/data_manifest/train.jsonl`, copied to
  `/dev/shm/xliu253_scheduled_unirope_20261104/` (sha256 `f8e68572…`) with its byte offsets.
  - 200k train sets: DL3DV 100k (perspective 288×512), HM3D 60k and JHU 40k (panorama 256×512). The 10 held-out JHU scenes
    are excluded.
  - Each row carries `near_depth` from Zhaoliang's tables (`/cis/home/zzhan288/RayCrafter/near_depth/`), and preflight
    checks every row against them. UniRoPE doesn't use it.
- **Same rows as the concat run's manifest.** The same IDs in the same order, with identical poses and intrinsics. Only the
  image root and the `near_depth` fields differ.
- **Images.** PNGs under `/export/io104a/data/xliu253/omnicrafter_mixed_local_cache`, read through
  `image_path_map.shared.json`. On io105 that is NFS from io104.
- **Loading.** `dataset.py` reads each set inline on the training process (`num_workers=0`), which costs ~0.26 s per step:
  1. seek to the row and parse it;
  2. load the 2 reference and 3 ground-truth PNGs plus the 3 renders, with a PIL bilinear resize (no crop);
  3. build the CPU ray map.
- **Order.** `shuffle=True` with a generator seeded 20261023, sharded over 4 ranks by accelerate, so one pass is 50k steps.
  - This is the concat run's order: continuation step s reads the set the concat run read at step s − 50000.
  - The io104 SCoPE-FF run sees the same sets and tasks.
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
- `io105_launch/launch_now.py`: `ROOT` is xliu253's directory, and it allows only the four PIDs above.

`queue_training.py`, `launch_now.py` and `train.py` refuse an output that already has checkpoints. Pointed at xliu253's run,
they would stop there.
