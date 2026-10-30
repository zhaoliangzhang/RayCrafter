# Mixed-projection checkpoint 50,000: matched evaluation on io95

Run `bash run_io95.sh` on io95. The controller uses GPU UUIDs corresponding to
indices 0–6 only. GPU 7 belongs to another job and is never selected or stopped.
All seven selected GPUs must be idle before each GPU phase starts. While they
are occupied, the controller polls every 60 seconds without a time limit; it
requires three idle checks 15 seconds apart before launch. CPU tests and input
validation can run while waiting. A dedicated tmux session keeps the controller
alive after disconnecting. No existing GPU process is stopped or shared.

The reference benchmark is the completed step-22,000 evaluation. We copy its
manifest byte-for-byte, preserve its per-set seeds, and reverify every saved
input-image checksum. The model/camera code snapshot is unchanged. No predictions
from the old checkpoint are reused. Existing reference, ground-truth, and cached
point-render files are linked into new visual bundles; the prepared metric GTs
from the previous run are reused without another resize.

## Protocol

- Checkpoint: mixed-projection `step-50000.safetensors`.
- 2 references + 3 jointly generated targets; 20 steps, CFG 1, sigma shift 5.
- Paired initial noise for NVS and cleanup. No evaluation-time pixel dropout.
- HM3D: 423 sets / 1,269 targets per task; Replica: 68 / 204; JHU: 500 / 1,500.
- DL3DV perspective: 2,800 sets / 8,400 targets per task.
- Panorama: 512×256 (width×height); perspective: 512×288.
- NVS: all-black query images. Cleanup: the same cached PanoVGGT panorama renders
  and supplied VGGT-Omega perspective renders used in the previous evaluation.
- PSNR, SSIM, LPIPS-VGG, DreamSim ensemble distance, and standard FID via Clean-FID
  clean mode / 2048-D Inception features. Dataset and pooled category summaries.
- Retain all original HM3D groups, and additionally report strict metrics excluding
  the same 21 reference/target overlaps. Do not resample or alter inference groups.

Output:
`/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/evaluations/step50000_dl3dv_hm3d_replica_jhu_q3_20261028`

Inspect `runtime/status.json`, `logs/`, and `bundles/`. Final aggregate results
will be written to `metrics/summary.json` and `metrics/summary.csv`, followed by
the root `_SUCCESS` marker. The controller performs CPU tests, inference and
metric smoke tests, seven-way inference, seven-way scoring, and final aggregation.
Rerunning the same controller preserves completed outputs and resumes unfinished
inference shards. `prepare.py` is the retained original preparer for provenance;
this run uses `prepare_matched.py` instead.
