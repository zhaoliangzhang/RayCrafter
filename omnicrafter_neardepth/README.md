# Near-depth mixed-projection SCoPE (portable)

This is xliu253's `mixed_neardepth_2ref3_20261029` experiment with the launch made portable for
Skipjack. On io100 it trains as
`/cis/net/io81/data/xliu253/omniraycrafter/mixed_scope_neardepth_q3_dl3dv100k_hm3d60k_jhu40k_b1_100k_io100`.

| Path | What it is |
| --- | --- |
| `snapshot/` | `/cis/home/xliu253/Documents/PanoLagerNVS/mixed_neardepth_2ref3_20261029` as of 2026-10-29 15:43 EDT, unmodified (132 files, `__pycache__` excluded). Its io100 launchers (`queue_io100.py`, `run_io100.sh`, `stage_wan.py`, `prepare_cached_manifest.py`, `configure.py`) are kept for provenance and do not run elsewhere. |
| `near_depth/` | The two per-scene near-depth tables the io100 run uses: Zhaoliang's `near_depth/dl3dv_points.json` (`perspective.json`) and `framecrafter_5input.json` (`panorama.json`). The hashes match the run's `data_manifest/summary.json`. |
| `tools/prepare_manifest.py` | Portable `prepare_cached_manifest.py`. It rebuilds the run's 200k-set manifest from `omnicrafter_handoff/metadata/train_cached.jsonl.gz` at your cache root. |
| `tools/train_neardepth.py` | Portable `queue_io100.py`: the same input checks, the 36-channel SCoPE initialization, the CPU preflight (the snapshot's three test files and `check_data.py`), then one 4-GPU `accelerate` launch of the snapshot's `train.py`. |
| `tools/entry.py` | Points the snapshot's `settings` / `common` at the config's paths, then runs `train.py` or `check_data.py` unchanged. |
| `configs/` | `train.example.json` and the `skipjack_train.sbatch` template. |

## Recipe

These are fixed by the snapshot, not by the config.

- **Data:** 2 refs + 3 queries. 200k sets: DL3DV 100k, HM3D 60k, JHU 40k. Perspective 512×288 and panorama 512×256 batches are half each.
- **Tasks:** NVS per sample with p 0.5 (all three queries black together). Clean-up queries drop 0–100% of pixels.
- **Model:** Wan2.1-I2V-14B with the FrameCrafter LoRA. The patch embedding is cut to its 36 image channels, so there is no ray concat. Q/K-only SCoPE (400 tensors) is added after the Q/K norm and RoPE.
- **Pose units:** the third query is the origin with identity rotation. Translations are divided once by the scene's near depth (`third_query_frame_per_scene_near_depth_v1`).
- **Optimizer:** AdamW, lr 1e-5, weight decay 0.01, `ConstantLR` defaults, gradient clipping 1.0.
- **Scale and precision:** ZeRO-2 bf16, 4 GPUs × batch 1, gradient checkpointing.
- **Frozen encoders:** the empty-prompt T5 context is cached, and T5 and CLIP are freed.
- **Length:** 100k steps, checkpoint every 2k.

On io100 it uses 35.4 GiB peak per GPU and about 4.3 s per step.

## Steps on Skipjack

1. Transfer the assets, following `omnicrafter_handoff/docs/MIGRATION.md` and `docs/SKIPJACK.md`:
   - the RGB training cache (`omnicrafter_mixed_local_cache`, about 747 GB);
   - Wan2.1 (`Wan-AI/Wan2.1-I2V-14B-480P`, plus the umt5-xxl tokenizer under `Wan-AI/Wan2.1-T2V-1.3B`);
   - the published `framecrafter.safetensors` (sha256 `fa22ffdb…5b92524`).

   Restore `omnicrafter_handoff/metadata/train_cached.jsonl.gz`, as described in `OMNICRAFTER_HANDOFF.md` at the repo root.
2. Build the manifest. It takes about 1 min; add `--check-files` to also stat all 1.6M RGB files.

   ```bash
   python omnicrafter_neardepth/tools/prepare_manifest.py \
       --cache-root /scratch/jhu/<PI>/<USER>/omnicrafter/train_cache \
       --output /scratch/jhu/<PI>/<USER>/omnicrafter/neardepth_manifest
   ```

   It stops unless the result equals io100's manifest apart from the cache root (`canonical_sha256`).
3. Copy `configs/train.example.json` to a local file and set the five paths. Then run

   ```bash
   python omnicrafter_neardepth/tools/train_neardepth.py --config train.local.json --dry-run
   ```

   That checks the inputs and creates nothing. `--preflight-only` also makes the output directory and runs the CPU preflight.
4. Fill in `configs/skipjack_train.sbatch` (account, partition, `REPO_ROOT`, `TRAIN_CONFIG`, `TRAIN_PYTHON`) and submit it.
   - A new output runs the preflight first.
   - **Resubmitting the same job continues** from the newest `step-*.safetensors`. The adapter weights and step are restored; the optimizer, scheduler and data order restart, as they do on io100.
   - A job cut at the walltime loses at most the steps since the last 2k checkpoint. Chain jobs with `sbatch --dependency=afterany:<jobid>`.
   - Once `final_step` is saved, a resubmission only reports completion.

The launcher refuses to reuse an output whose config, snapshot code, tools or manifest differ from the ones that started it. Change anything and use a new `output`.

The environment needs the snapshot's stack: torch + CUDA, accelerate with DeepSpeed, OpenCV, safetensors and PyYAML. It was tested on CIS with torch 2.8.0+cu128, accelerate 1.14.0 and deepspeed 0.19.5.

**Smoke test (CIS io96, 4× A40, 2026-10-30):** the manifest was built at io100's cache root and came out byte-identical to io100's.
- **Run:** `final_step` 20, `save_every` 10. The preflight passed and training ran at 5.2 s/step with 35.4 GiB peak per GPU (io100: 35.4 GiB).
- **Interruption:** a SIGTERM after `step-10` stopped it cleanly. Relaunching resumed from step 10 and finished at step 20. A further launch only reports completion.
- **Startup:** loading Wan from NFS took about 14 min the first time (the first step came 15 min after the launch) and about 3 min from the page cache.
