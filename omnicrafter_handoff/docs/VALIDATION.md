# What was verified, and what remains

Original validation date:2026-10-28. No GPU training/inference was started for packaging,
and no production source, data manifest, image or existing job was modified.

## SCoPE portable training update: 2026-10-29

The original package bundled SCoPE source but its portable launcher hard-coded
concat. This folder revision adds explicit architecture selection, a separate
CPU 420-to-36 published-adapter conversion, trained-SCoPE checkpoint validation,
the SCoPE example JSON and Slurm instructions. The model implementation itself
is unchanged. No training/evaluation job was launched by this fix.

Checks executed on the source system with CUDA hidden:

| Check | Result |
| --- | --- |
| Portability tests, CPU conversion and opt-in original-mount preflights |25 passed |
| Frozen mixed camera/loss tests, including small-model SCoPE forward/backward and alternating projection grids |12 passed; 1 CUDA-only test skipped |
| Real published adapter conversion |All 802 tensor values match the prescribed slicing/preservation; source SHA unchanged; overwrite rejected |
| Real trained SCoPE checkpoint preflight |36-channel patch and all 400 camera parameter names/shapes accepted |
| Bundled source/data identity |477 files under snapshots, metadata, panorama and panovggt match the original package hashes |
| Training Slurm/generic shell syntax |Passed |

The launch-dispatch test mocks Accelerate: it proves the effective config and
SCoPE arguments are passed correctly, not that a full 14B model fits GPUs.
The small-model forward/backward test is real CPU tensor computation. A full
four-rank GPU smoke on Skipjack remains required. SCoPE evaluation is explicitly
unsupported by the portable evaluator; no SCoPE inference parity is claimed.

To repeat the source-only checks with access to the original mounts:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OC_TEST_TORCH=1 OC_VERIFY_ORIGINAL_MOUNTS=1 \
  python -B -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -B snapshots/mixed/test_protocol.py
```

On a new cluster, omit `OC_VERIFY_ORIGINAL_MOUNTS`; those four tests will skip.
The original October 28 archive is unchanged and does not contain this fix.
`PACKAGE_FILES.json` has been regenerated for the updated folder.

## Executed checks

| Check | Result |
| --- | --- |
| New portability/helper tests |12passed |
| Read-only preflight against original data/weight mounts |2passed |
| Frozen mixed camera/loss protocol tests |12passed;1CUDA-only test skipped |
| Existing portable panorama package tests |18passed |
| Existing two-reference PanoVGGT package tests |14passed |
| Slurm and generic shell template syntax (`bash -n`) |Passed |
| Full training/evaluation metadata audit |200,000 training entries;3,791 evaluation groups; expected counts/seeds |
| JHU held-out sequence exclusion |All10 excluded from current mixed training |
| Evaluation input transfer plan |43,678 deduplicated assets, including metadata; no copying performed |

The full metadata audit is recorded in `provenance/manifest_audit.json`.
The current cached training manifest's uncompressed SHA256 is:

`1020aff9922494cc44ea9204740faab0814ac0b823ed0aac404d6ef81f04a184`

Its source original-path mixed manifest SHA256 is:

`b5b02c967c6f1ec210980ec36a7c89bd34d68a877cf5530c4e16dcc5e4c8a71c`

The frozen evaluation manifest SHA256 is:

`e9701d0a435645638cd69f11a76b9c802892309c92f206da7b5449afc7162536`

`PACKAGE_FILES.json` records every distributed file's size and SHA256. It is
generated after documentation/code edits and excludes itself. Run
`python tools/package_inventory.py` on the untouched extracted archive to check it.

## Exact scope of the new code changes

The `snapshots/` source is copied, not refactored. New wrappers bind filesystem
locations, output paths, cumulative step/save interval and inference shard count
before importing the existing implementation. The mixed training wrapper accepts
explicit concat or SCoPE architecture, locks batch/GPU count and requires the known manifest mix;
it disables the historical opt-in GPU-memory-parking path. Its math/loss/task
mixture are the source implementation, not a newly invented model.

Path migration is explicit prefix mapping, not a global string substitution.
It preserves sample identity/order, camera data, projection, split and seeds.
The training index is rebuilt from the rewritten byte stream, and synthetic
tests seek both rows successfully using the new offsets. A second write to the
same manifest is rejected. Evaluation content hashes can be verified separately
after the real assets have been transferred.

## Not yet validated

- Fresh installation on Skipjack, CUDA extension compatibility or GPU memory fit.
- End-to-end GPU execution of the new location-binding launchers on a destination allocation.
- Numeric equivalence of generated images/perceptual metrics across different GPU/software stacks.
- Availability, ownership, billing allocation or capacity of your destination storage/GPUs.
- Transfer completeness on a new filesystem; `verify_assets.py` / rsync checksum checks must run there.
- A regenerated PanoVGGT prior matching an old cached prior byte-for-byte.
- Full optimizer-state resume: current checkpoints do not contain that state.
- Physical/geographic disjointness beyond the documented split units; matching
  sequence-name tests are not proof that two captures never depict the same place.

The source metric regression and source cache-ready report are historical
evidence, not substitutes for these destination checks. The current step50k
evaluation is not labeled complete or assigned final metrics by this package.

## Minimal acceptance before a large new-cluster run

1. Verify the archive; transfer real files; check data hashes and rebuilt offsets.
2. Confirm the expected 200k training composition and JHU exclusions.
3. Run the CPU tests in the intended environments.
4. Submit a small four-GPU training smoke to a new output directory; check both
   projection sizes, both tasks, finite loss and CUDA optimizer state.
5. Run `tools/evaluate.py --phase smoke` to test one set per dataset on an
   allocated GPU; inspect all24 predictions and the reference/GT/prior links.
6. Confirm checkpoint keys/shapes and metric environment/cache versions before
   committing to a long production run or publishing migrated results.
