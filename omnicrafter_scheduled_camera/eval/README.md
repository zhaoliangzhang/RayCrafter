# Step-100000 evaluation of the io105 UniRoPE v3 run

xliu253's evaluation of this run's `step-100000.safetensors`, copied unchanged from
`/cis/home/xliu253/Documents/PanoLagerNVS/`:

- **`scheduled_unirope_eval_100000_20261105/`**: the evaluation itself.
  - `run.py` is the controller. `prepare.py` builds the benchmark folder and `worker.py` runs inference.
  - `score.py` and `aggregate.py` compute the metrics through `engine/`. `test_eval.py` holds the CPU tests.
  - `training_snapshot/` is the trainer, identical to `../`. The worker imports its camera code from there.
- **`panorama_nvs_transfer_20260818/workspace/baselines/replica_2ref3/score_predictions.py`**: the PSNR/SSIM/LPIPS functions
  `score.py` imports. It sits at the same path relative to the evaluation folder as in xliu253's tree.
- **`results_step100000/`**: the evaluation's `provenance.json` and `summary.json` / `summary.csv`.

All 146 files hashed in `results_step100000/provenance.json` (`code_sha256`) match these copies byte for byte.

## What it does

- **Benchmark.** The concat step-50000 evaluation's manifest, byte for byte:
  - HM3D 423, Replica 68 (204 targets), JHU 500 held-out and DL3DV 2800 test sets.
  - Each set has 2 references and 3 targets, at 512×256 (panorama) or 512×288 (perspective).
- **Tasks.**
  - NVS: all three query inputs are black.
  - Cleanup: the cached priors are used without pixel dropout (PanoVGGT renders for panoramas, the supplied DL3DV priors).
- **Sampling.** 20 steps, CFG 1, sigma shift 5, seed 20260904 + set index. Each set's noise hash is checked against the
  concat step-50000 evaluation.
- **Model.** FrameCrafter's Wan pipeline with 420 input channels, plus the trainer's `install_camera_attention` and
  `model_fn`.
  - All 882 checkpoint tensors are loaded and compared after loading.
  - The gate is 1.0, read from the checkpoint's schedule metadata.
- **Cameras.** The trainer's `ordered_cameras` and `prepare_camera_context`: q3 frame, scaled so the two references are 1
  apart. The benchmark's near-depth file is kept for provenance only; UniRoPE doesn't use it.
- **Metrics.** PSNR, SSIM, VGG LPIPS, DreamSim and Clean-FID (Inception 2048-D).
  - `HM3D_strict` leaves out the 21 target views that overlap a reference.
  - An audit hook stops inference from opening any target image.

## How it ran

On io95's 8 A40s with 32 GiB of weights cached on each GPU. Inference started on 2026-11-05 at 23:55 and metrics finished on
2026-11-06 at 13:47. Inference took about 47 s per set-task per GPU.

## Results

PSNR, NVS / cleanup, from `results_step100000/summary.csv`. The concat column is the step-50000 model this run started from,
on the same benchmark.

| | UniRoPE v3 step 100000 | concat step 50000 |
|---|---|---|
| DL3DV | 18.72 / 19.85 | 20.37 / 21.03 |
| HM3D | 23.26 / 24.61 | 24.24 / 24.94 |
| Replica | 25.08 / 26.55 | 26.25 / 27.05 |
| JHU | 18.43 / 18.57 | 19.18 / 19.05 |

## Running it elsewhere

Not adapted: everything machine-specific is set in `scheduled_unirope_eval_100000_20261105/eval_shared.py`.
- `CHECKPOINT` and `OUT` point at the run's io81 folder.
- `MATCHED`, `COMBINED` and `PREVIOUS` are earlier evaluations it reads from: the manifest, near depth, input hashes, seeds
  and the metric environment.
- `HOST`, `WAN`, `CACHE_ROOT`, the DreamSim and FID sources, `FID_PACKAGES`, `PYTHON` and `METRIC_PYTHON` are xliu253's
  machines and environments.
- `GPUS`, `NUM_SHARDS` and `VRAM_LIMIT_GIB` set the GPUs, shard count and per-GPU weight cache.

The controller checks the hostname and GPU UUIDs. Every script checks the code hashes against the output folder's
`provenance.json`, so an edited copy needs a new output folder. `VRAM_LIMIT_GIB` is the per-GPU weight cache; on 80 GB H100s
it can be large enough to keep the whole model on the GPU.
