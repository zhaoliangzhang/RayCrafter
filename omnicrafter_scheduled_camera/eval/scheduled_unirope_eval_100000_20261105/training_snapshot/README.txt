Scheduled camera continuations -- 2026-11-04
===========================================

Two separate experiments, NOT the old concat+SCoPE-FF+UniRoPE combination:
  scope_ff: concat + scheduled SCoPE-FF, io104 GPUs 0-3.
  unirope:  concat + scheduled UniRoPE v3, io105 GPUs 0-3.

Initialization (immutable, no teacher updates):
  /cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/step-50000.safetensors
Strictly restore all 802 inherited LoRA/420-channel patch tensors. New camera
modules are separately initialized. These runs do NOT restart from the
published FrameCrafter adapter and do NOT load the old combined checkpoint.

Each run performs 300,000 additional optimizer updates, global steps
50,000 -> 350,000. Save step-52000.safetensors, then every 2,000 steps.
Linear injection schedule: lambda = clamp((step-50000)/10000, 0, 1).
Before training lambda=0; first update uses lambda=0.0001; at 60000 it is 1.
Standard inherited LoRA32 and patch weights remain trainable throughout,
LR=1e-5, weight decay=0.01. No extra base-WAN layers are unfrozen.
The exact old checkpoint output is guaranteed only before optimizer updates;
later, lambda=0 recovers the THEN-CURRENT concat branch, not frozen 50k weights.
New SCoPE-FF LR=1e-5. UniRoPE depth LR=4e-4, weight decay=0.5.

Camera conventions:
  Model order stays [ref1,ref2,q1,q2,q3]; no reference sorting.
  Global frame for ALL branches: inverse(T_q3) @ T_i (camera-to-world).
  q3 is the origin with identity rotation, regardless of noisy/black pixels.
  Concat rays retain original mean-distance scaling and inverse OpenCV
  distortion. This is required to preserve the trained 50k concat pathway.
  SCoPE-FF translations use per-scene near-depth, not mean-distance scale.
  UniRoPE uses unit reference-baseline scale, with the shared q3 world gauge.
  UniRoPE's internal per-camera relative-attention passes remain intact;
  they are not independent sample normalization or different output poses.
  New attention rays retain upstream pinhole-K-only perspective rays.

SCoPE-FF keeps WAN's original spatial RoPE and zero temporal phase:
  Q = Q_baseline_rope + lambda * SCoPE_Q(camera)
  K = K_baseline_rope + lambda * SCoPE_K(camera)
  V unchanged. Gate is AFTER SCoPE normalization. Final MLP layers zero-init.
UniRoPE blends whole attention outputs before the inherited output projection:
  A = (1-lambda)*A_original_spatial_rope + lambda*A_UniRoPE
  Hard bypass at zero; only UniRoPE at one. During the transition two attention
  paths are computed, not two full WAN networks. No SCoPE modules in this run.

Near-depth source files from RayCrafter TRAINING_NOTES.md:
  /cis/home/zzhan288/RayCrafter/near_depth/dl3dv_points.json
  /cis/home/zzhan288/RayCrafter/near_depth/framecrafter_5input.json
preflight.py checks every manifest row against the matching scene in these
files, verifies their hashes, and archives copies in OUTPUT/initialization/.

Training data: existing 200,000-set mixed manifest, no sample rewriting:
  /cis/net/io81/data/xliu253/omniraycrafter/mixed_rayrope_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/data_manifest/train.jsonl
DL3DV=100000, HM3D=60000, JHU=40000. Train split only; JHU ten heldout
scenes excluded. Same source order, IDs, images, GT, poses and noisy renders.
Images stay on io104a; image_path_map.shared.json resolves the shared mount.
Only metadata is copied into each host's /dev/shm, not the large image dataset.
Panorama width x height 512x256; perspective 512x288.

Task mixture: one Bernoulli(0.5) draw PER SAMPLE.
  NVS: all three query images black together.
  Cleanup: existing noisy renders, independent per-image pixel dropout
           uniformly chosen in integer percentages 0..100 inclusive.
Redraw accidental all-three-black cleanup augmentation. Entirely empty
source-render sets are replaced with another train-only set, same scene
preferred, same dataset/projection/resolution mandatory. Task and manifest
are never silently changed. Existing dataset regression tests cover this.

Hardware: four GPUs, batch size 1/GPU, global batch 4, accumulation 1,
BF16, CPU model initialization, ZeRO2, GPU AdamW. No optimizer/parameter/
encoder/activation CPU offload. Gradient checkpointing on. 10M buckets.

Queue (persistent tmux, no expiry, no preemption, no memory holders):
  On io104: bash queue_scope_io104.sh
  On io105: bash queue_unirope_io105.sh
Poll every 60 seconds; require all four cards to have no compute PIDs,
<=1024 MiB used and <=5% utilization for three consecutive checks.
CPU preflight runs while waiting. Inputs/code are hash-checked again before
launch. On training failure, preserve results and report failed; do not loop
crash/restart or kill another job. A disconnect does not stop tmux.

Correctness, not pilot studies:
  CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 CAMERA_VARIANT=scope_ff \
    /cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -m unittest test_scheduled test_dataset_resume test_queue -v
  Repeat with CAMERA_VARIANT=unirope.
After GPUs become idle, three tiny distributed GPU correctness updates check
compiled BF16, both real grids and GPU optimizer state. No experiment weights
are used or saved by that check. Actual 14B gate-zero forward parity is checked
against original concat attention on both projections/tasks before first update.
Failure blocks training. No separate pilot experiment is launched.

Monitoring files in each OUTPUT (exact paths in config.py):
  queue_status.json, preflight.json, launch.json, progress.json,
  train_loss.jsonl (loss/gate/task ratios every 10 steps), logs/,
  gate_zero_full_model_rankN.json, gpu_correctness_rankN.json,
  gpu_optimizer_rankN.json, step-N.safetensors.
Checkpoints record architecture, coordinate frames, source, schedule, step and
gate. Explicit --restore-checkpoint restores weights and schedule; optimizer
moments and RNG reset, as with prior adapter-only checkpoint recipes.

Do not edit code while a queue is active: source fingerprints prevent stale
queued jobs from silently launching changed experiments. Existing evaluation
and training folders are untouched.
