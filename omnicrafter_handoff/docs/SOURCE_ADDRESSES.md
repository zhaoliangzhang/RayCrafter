# Source addresses (old cluster; not portable configuration values)

These addresses identify data provenance. On the destination cluster use your
own roots and the relocation tools. A path under `/export/<server>/...` is local
to that server; `/cis/net/<server>/...` may be an NFS alias. Do not assume the two
work on every host or survive relocation.

## Original and processed data

| Dataset / layer | Address |
| --- | --- |
| HM3D processed training source 1 | `/cis/net/io80/data/xliu253/HM3D_Bi_data_train_1` |
| HM3D processed training source 2 | `/cis/net/io81/data/xliu253/HM3D_Bi_data` |
| Replica processed training source | `/cis/net/io84a/data/xliu253/Replica_Bi_train` |
| JHU original unprocessed JPEGs | `/cis/net/io99a/data/xliu/JHU_ULTRA_360/orig` |
| JHU scene-split clean RGB | `/cis/net/io99a/data/xliu/JHU_ULTRA_360/dataset/images` |
| JHU scene-split pose metadata | `/cis/net/io99a/data/xliu/JHU_ULTRA_360/dataset/metadata` |
| JHU processed RGB/renders/point maps | `/cis/net/io84a/data/xliu253/JHU_Bi_data` |
| Perspective prepared train | `/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/train` |
| Perspective prepared test | `/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test` |
| HM3D validation scene GLBs | `/cis/net/io72a/data/xliu253/HM3D_val/hm3d-val-glb` |
| HM3D validation Habitat assets | `/cis/net/io72a/data/xliu253/HM3D_val/hm3d-val-habitat` |
| HM3D validation PointNav episodes | `/cis/net/io89/data/xliu253/HM3D/pointnav/val` |

The raw meshes/episodes are needed to regenerate imagery, not to run the already
prepared training/evaluation datasets in this package.

## Common panorama training copy

Root:
`/cis/net/io80/data/xliu253/framecrafter/5input_model`

Important paths relative to that root:

| Purpose | Relative path |
| --- | --- |
| HM3D clean/noisy RGB | `HM3D/` |
| Replica clean/noisy RGB | `Replica/` |
| JHU clean/noisy RGB | `JHU/` (including `samples_bottom_person`) |
| All poses, OpenCV w2c | `poses/manifest.json`, with NPZ files below `poses/framecrafter/` |
| Legacy all-dataset train | `train.jsonl` (`train.json` is an alias) |
| Legacy all-dataset val | `val.jsonl` (`val.json` is an alias) |
| HM3D-only 2ref3 train | `manifests/hm3d_train.jsonl` |
| Replica-only 2ref3 train | `manifests/replica_train.jsonl` |
| HM3D one-query cleanup train | `manifests/hm3d_train_cleanup_q1.jsonl` |
| JHU exclusion list | `manifests/jhu_holdout10_seed20260822/heldout_scenes.json` |
| JHU train without 10 sequences | `manifests/jhu_holdout10_seed20260822/jhu_train_2ref1_excluding_10_scenes.jsonl` |
| Larger JHU held-out source pool | `manifests/jhu_holdout10_seed20260822/jhu_test_2ref1_10_scenes.jsonl` |
| Historical balanced 2ref1 train | `manifests/hm3d_jhu_2ref1_balanced_jhu_holdout10_seed20260903/train.jsonl` |
| Original FrameCrafter 2ref1 manifests | `manifests/framecrafter_original_2ref1/` |

Historical 1:1 panorama **2ref3** training manifest:

`/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/data_manifest/train.jsonl`

## Current mixed train: recommended transfer source

Original no-copy manifest directory:

`/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/data_manifest`

Consolidated, checksum-verified image cache:

`/cis/net/io104a/data/xliu253/omnicrafter_mixed_local_cache`

Local alias **on io104**:

`/export/io104a/data/xliu253/omnicrafter_mixed_local_cache`

Transfer `images/` from this cache, not each source server separately. It contains
1,361,608 unique files: DL3DV 781,656; HM3D 445,911; JHU 134,041. Verified payload:
747,172,624,878 bytes. The cache's `data_manifest/train.jsonl` embeds the local
`/export/io104a/...` prefix and needs relocation. Our archive includes that file
as `metadata/train_cached.jsonl.gz`; its uncompressed hash is in the audit.

The cache was verified complete on 2026-10-27. That fact alone does not mean any
currently running job was switched to it; migration and trainer restart are
separate actions.

## Evaluation inputs

| Identity | Root / manifest |
| --- | --- |
| New HM3D | `/cis/net/io72a/data/xliu253/hm3d_val_data` |
| New Replica204 | `/cis/net/io72a/data/xliu253/replica_val_data` |
| JHU held-out500 | `/cis/net/io80/data/xliu253/framecrafter/5input_model/evaluation/jhu_holdout10_500_2ref3` |
| Perspective test | `/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test/benchmark/manifest.json` |
| Cached raw panorama PanoVGGT priors | `/cis/net/io81/data/xliu253/framecrafter/evaluations/request_20260904/panovggt/q3_grouped/samples` |
| Perspective test priors + metadata | `/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test/priors/samples` |
| OLD Replica1000 baseline inputs | `/cis/net/io81/data/xliu253/framecrafter/baselines/replica_2ref3_256x512` |
| OLD Replica1000 baseline outputs | `/cis/net/io81/data/xliu253/framecrafter/baselines/replica_2ref3_256x512_results` |

Each new panorama root has `val_2ref3_nvs.jsonl`, `val_2ref1_nvs.jsonl` and a pose
manifest. HM3D/Replica also have 2ref10 and precomputed point-drop variants.
For migration of the **current matched evaluation**, use the self-contained
`metadata/eval_manifest.json` rather than rebuilding these groups from scratch.
Its paths include pre-resized metric GTs in the old step22k evaluation directory;
the transfer planner includes those files too.

## Models / code / prior results

Current mixed training code:
`/cis/home/xliu253/Documents/PanoLagerNVS/mixed_projection_2ref3_20261023`

Current mixed checkpoint root:
`/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104`

Its evaluation directories:

- `evaluations/step22000_dl3dv_hm3d_replica_jhu_q3_20261024` (previous matched evaluation).
- `evaluations/step50000_dl3dv_hm3d_replica_jhu_q3_20261028` (resumed evaluation; not assumed complete by this package).

Panorama teacher:
`/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/step-650000.safetensors`

Teacher evaluation reference:
`/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/evaluations/step650000_hm3d_replica_jhu_q3_256x512_20260917`

Published FrameCrafter initialization used here:
`/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/weights/framecrafter_released/framecrafter.safetensors`

WAN local evaluation cache on io95:
`/export/io95/data/xliu253/framecrafter_wan21`

Shared WAN source:
`/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/checkpoints/wan21`

DreamSim source cache on io95:
`/export/io95/data/xliu253/oc_teacher_student_benchmark_cache/dreamsim`

FID weight source:
`/cis/net/io96/data/xliu253/oc650k_eval_cache/fid/inception-2015-12-05.pt`

LPIPS/VGG and other torch-hub downloads must also be staged from the metric
environment's effective `TORCH_HOME` (or downloaded before an offline job).
Do not copy only WAN and assume perceptual metrics are self-contained.
