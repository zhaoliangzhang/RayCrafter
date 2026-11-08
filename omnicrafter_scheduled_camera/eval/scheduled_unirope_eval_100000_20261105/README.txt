Scheduled concat + UniRoPE, global checkpoint 100000, full validation

Run on io95 GPUs 0-7 using bash run_io95.sh inside tmux.
The launcher waits for idle GPUs and never stops unrelated work.
Training on io104/io105 is not modified.

Inputs are byte-identical to the completed concat-only 50k benchmark manifest:
423 HM3D sets, 68 Replica sets (204 target images), 500 JHU held-out sets,
2800 DL3DV test sets. Each set has two references and three jointly generated
targets. Panorama W x H = 512 x 256; perspective = 512 x 288.
NVS uses all-black target inputs. Cleanup uses the previously verified cached
PanoVGGT panorama priors / supplied DL3DV priors, with no pixel dropout.
Reference/GT/original-GT/prior symlinks are placed beside predictions.

20 denoising steps, CFG=1, sigma shift=5, original per-set seeds preserved.
Checkpoint-trained camera order/gauge is preserved: ref1,ref2,q1,q2,q3;
q3 full SE3 origin; mean-distance concat scale and unit-baseline UniRoPE.
Injection gate is the saved 1.0 after a 10k ramp from global step 50000.
Original 72k evaluation camera/training code is snapshotted without modification.
The 100k checkpoint is verified tensor-for-tensor after loading all 882 tensors.

Metrics: PSNR, SSIM, VGG LPIPS, official ensemble DreamSim, Clean-FID clean
mode with Inception 2048-D. No prediction resizing. GT uses the same cached
bilinear-resized images as all previous evaluations. Metric smoke test
recomputes baseline predictions and checks historical scores.
HM3D all-target results preserve 21 reference-overlapping target views;
extra strict scores exclude those views. Replica 204 FID has small-sample bias.

The controller performs CPU tests, local WAN checksum verification, four-
dataset inference/metric smoke checks, eight resumable inference shards,
eight metric shards, and final aggregation. Completed predictions are not
overwritten. The controller stops only its own children if a check fails.
32 GiB device weight caching threshold with bounded inference offload; no
change to model weights, dtype, camera arithmetic or image resolution.
CPU model initialization is serialized with a 75 GiB available-RAM guard.

Outputs: checkpoint-parent/evaluations/
 step100000_hm3d_replica_jhu_dl3dv_q3_20261105/
  bundles/DATASET/TASK/SCENE/SET/predictions/
  metrics/summary.json, summary.csv, per_target.json
  runtime/status.json, provenance.json, logs/
