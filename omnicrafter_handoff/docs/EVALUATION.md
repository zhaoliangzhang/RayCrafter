# Exact evaluation contract

The portable `tools/evaluate.py` currently supports **concatenated-ray**
OmniCrafter checkpoints only. The October 29 update enables SCoPE **training**,
not SCoPE evaluation. An explicit `architecture: "scope"` or a SCoPE checkpoint
is rejected here. Do not evaluate a SCoPE checkpoint by changing only the input
channel count: its camera modules and token-level rays must also be installed.

The included `metadata/eval_manifest.json` is copied byte-for-byte from the
step50k evaluation, which itself preserves the step22k sample ordering. It is the
preferred source of truth for migrating the current mixed-model benchmark.
Path relocation may change its file hash; the original hash and mapping are
recorded, while image content, grouping, poses, seed and order must remain intact.

| Dataset | 2ref3 sets | Targets per task | Both tasks |
| --- | ---: | ---: | ---: |
| HM3D | 423 | 1,269 | 2,538 |
| Replica204 | 68 | 204 | 408 |
| JHU held-out10 | 500 | 1,500 | 3,000 |
| DL3DV perspective | 2,800 | 8,400 | 16,800 |
| Total | 3,791 | 11,373 | 22,746 |

## Conditioning and generation

- Two clean references and three simultaneously generated target slots.
- NVS: all three query condition RGB images are exactly black.
- Cleanup: cached raw point-cloud renders; no evaluation-time pixel dropout.
- Panorama output512×256 W×H; perspective output512×288 W×H.
- Teacher/multi-step sampler:20steps, CFG1, sigma shift5, empty positive/negative prompt.
- Seed =20260904 + frozen global set index; NVS/cleanup share the same initial noise.
- Reference/conditioning input resize is PIL bilinear; output predictions are **not** resized for scoring.
- Scores use the same pre-resized RGB metric GTs from the prior evaluation.

Changing seed/order, query grouping, projection resolution, sigma shift, denoising
steps, VAE tiling or attention backend can change results. A different GPU type
or torch/kernel stack can introduce numerical differences even with identical
seeds. Do not promise bitwise reproducibility across hardware.

## Cleanup priors: reference-only geometry

For HM3D, Replica and JHU: pass only the two references through PanoVGGT, align
predicted reference cameras/cloud to the two GT reference poses, and rasterize
that cloud at the three GT target poses. Target **poses** are allowed; target
RGB/depth must not be used for reconstruction, alignment or filling missing points.
The renderer retains black holes; no target-color completion is allowed.

For perspective: use the supplied VGGT-Omega priors/calibration and reference-pose
alignment, including its distortion convention. Do not apply the ERP rasterizer
to the perspective images. The supplied perspective preparation implementation
and its model checkpoint are **not** reproduced by the PanoVGGT subpackage; copy
the existing perspective priors to reproduce this benchmark.

The supplied `panovggt/README.md` explains similarity alignment, masks, degenerate
baselines and its optional fallback-scale calibration. If regeneration is needed,
read that contract and match the original prior metadata/checksums. The simplest
exact migration is to copy existing cached priors and `metadata.json` files.

## Metrics

- **PSNR:** RGB float32 in[0,1], per-target MSE, then average per-target PSNR.
- **SSIM:** scikit-image,11×11 Gaussian window, three RGB channels, data_range1.
- **LPIPS:** VGG network, officialLPIPS v0.1.4; no substitution with AlexNet.
- **DreamSim:** official pretrained ensemble distance, its own preprocessing.
- **FID:** Clean-FID clean mode,2048-D Inception features; computed over the whole
  dataset, not per-image and not the mean of shard FIDs.

PSNR/SSIM higher is better; LPIPS/DreamSim/FID lower is better. Pooled panorama
FID is computed from pooled features; it is not an average of HM3D/Replica/JHU FID.
Replica204 is small and correlated, so its FID is especially sample-size-sensitive.

Older panorama experiments also report WS-PSNR (ERP latitude-weighted error).
The shared scalar helper still computes it, but the **current frozen mixed
evaluation's report exports PSNR/SSIM/LPIPS/DreamSim/FID**, not WS-PSNR. It should
not be applied as an ERP metric to perspective images without redefining it.

The historical metric environment is separate from the training environment:

| Component | Observed metric version |
| --- | --- |
| torch / torchvision |2.4.0+cu118 /0.19.0+cu118 |
| numpy |1.26.4 |
| scikit-image |0.25.2 |
| LPIPS / DreamSim |0.1.4 /0.2.1 |
| Clean-FID |0.1.35 |
| timm / peft / transformers |0.6.13 /0.13.2 /4.44.2 |

`provenance/source_metric_regression.json` records a successful original smoke
regression against the prior metric implementation. It is **not** a new-cluster
validation result. Inception checkpoint SHA256:

`f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4`

## Reports and linked bundles

The wrapper keeps the existing layout:

```text
OUTPUT/
  benchmark/manifest.json
  provenance.json
  portable_request.json
  bundles/DATASET/{nvs,cleanup}/SCENE/SET/
    references/ref_0.png, ref_1.png       links to original refs
    ground_truth/query_0.png ...         links to exact metric GT
    ground_truth_original/query_0.png ...links to original GT
    noisy_inputs/prior links ...         cleanup only, plus provenance
    predictions/query_0.png ...          generated outputs
    _SUCCESS                            completion + seed/hash/timing/memory
  logs/                                 timestamped per-rank logs
  runtime/portable_status.json
  metrics/summary.json, summary.csv, per_target.json
  _SUCCESS                              whole evaluation completed
```

Completed set-task outputs are skipped on resume. Rerun with the **same config,
shard count, code, checkpoint and output directory**. If any protocol field
changes, use a new output directory. The controller fails if a worker fails;
it does not silently mark partial metrics as complete or automatically retry forever.
Metric scoring starts after inference and final aggregation follows all shards.

## Comparison pitfalls

1. Replica1000 and Replica204 are different benchmarks.
2. 2ref1/3/10 can share targets but differ in joint context and camera normalization;
   compare on the intersection of target identities and state the query count.
3. New HM3D exact includes21 reference-equal targets; strict excludes them only
   from scoring. Report which one is used.
4. A model previously trained on Replica/JHU has different exposure from a
   baseline that never used them. Report ancestry and held-out split units.
5. Cleanup using GT-depth-derived priors is not the same as cleanup using
   PanoVGGT-predicted priors. Never mix these labels in a results table.
6. PNG postprocessing or different GT resizing/perceptual network variants can
   change scores. Preserve the original metricGT files and caches.
7. Image prediction time recorded per set excludes model initialization; do not
   call it complete pipeline latency. PanoVGGT preprocessing time is separate.
