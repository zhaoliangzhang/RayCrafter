# OmniCrafter: cluster migration and experiment handoff

Snapshot date: 2026-10-28. Start here; do not infer an experiment from a folder
name or from a previous chat. This package is for transferring our **code,
experimental definitions and exact sample selection**, not for distributing the
large image collections or model weights. It does not launch anything by itself.

**Folder update, 2026-10-29:** portable **SCoPE training** is now wired up. Start
from [`configs/train_scope.example.json`](configs/train_scope.example.json) and
the [SCoPE launch instructions](docs/MIGRATION.md#scope-training).
The original October 28 `.tar.gz` has not been replaced; copy this updated
directory to obtain this fix. `PACKAGE_FILES.json` describes this folder revision.

## Recommended workflow

Use this versioned package plus an explicit JSON configuration per experiment.
Ask for a new command/configuration when the experiment changes, rather than
reconstructing the data split, conditioning and checkpoint ancestry each time.
Keep the configuration, source hashes, manifest hashes and metric versions with
every run. A command alone does not capture those facts.

Read these documents in order:

1. [Dataset definitions](docs/DATASETS.md): what our train/val/test names actually mean.
2. [Source address catalog](docs/SOURCE_ADDRESSES.md): original, processed, cached and evaluation locations.
3. [Model and experiment contract](docs/EXPERIMENTS.md): camera rays, task mixture, initialization and continuation.
4. [Transfer/setup/run instructions](docs/MIGRATION.md): commands and editable configurations.
5. [Evaluation protocol](docs/EVALUATION.md): exact benchmark, priors, metrics and comparability.
6. [Validation and limitations](docs/VALIDATION.md): what was tested and what still needs a new-cluster GPU smoke test.
7. [Skipjack-specific guide](docs/SKIPJACK.md): Slurm, VS Code lifetime, PI storage and editable batch templates.

## Included

| Directory | Purpose |
| --- | --- |
| `panorama/` | Previously shared portable panorama OmniCrafter implementation and README; supports historical panorama recipes |
| `snapshots/mixed/` | Frozen current mixed perspective/ERP training implementation, including its original cluster launchers for provenance |
| `snapshots/evaluation/` | Frozen step-50k evaluation implementation and the exact model snapshot it uses |
| `panovggt/` | Previously shared two-reference PanoVGGT renderer, its geometry tests and pinned model source |
| `tools/` | New path-relocation, transfer-list, integrity/audit and portable mixed train/eval wrappers |
| `configs/` | Editable path/run JSON examples and a **template**, not submitted, Slurm script |
| `metadata/` | Exact current 200k-set training manifest (gzip), exact 3,791-set evaluation manifest, input hashes and split summaries |
| `provenance/` | Original evaluation protocol, metric regression evidence and package validation records |
| `tests/` | CPU tests of the new portability helpers |

The mixed **training** wrapper supports `architecture: "concat"` (the default
for older configurations) or `architecture: "scope"` (explicit opt-in). Both
keep four GPUs, true batch one/GPU, two references and three targets, and the
same sample/task mixture. SCoPE changes camera injection and initialization,
not the dataset. It does not enable distillation or batch eight.

The portable **evaluation** wrapper is still **concat-only**, with configurable
independently sharded GPU workers. It explicitly rejects SCoPE rather than
silently using concatenated rays; SCoPE inference/evaluation needs a separate
adapter and has not been made portable by this update.

**Do not directly execute `queue_io104.py`, `run_io95.sh`, or other historical
launchers inside `snapshots/` on the new cluster.** They contain old host paths
and GPU identities. Use `tools/train_mixed.py` / `tools/evaluate.py` instead.

## Not included / not changed

- No RGB/depth image files, learned checkpoints, WAN backbone, environment binaries or results.
- No new-cluster submission, data transfer or existing-job interruption.
- No changed training samples, held-out scenes, ground truth, camera conventions or loss functions.
- No portable training implementation for SyncFix, FixAnything, baselines or distillation; those are distinct projects.
- No claim of bit-exact optimizer resume from our adapter-only `.safetensors` files.

The existing image cache is about **695.9 GiB** and must be transferred separately.
The gzip training manifest is included so the 200,000 selected entries, including
intentional perspective repeats, are not accidentally regenerated differently.

## Quick checks (no GPU, weights or original data mounts required)

From the extracted package, with Python 3.11:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
python tools/audit_manifests.py
python tools/package_inventory.py
```

After copying assets, installing environments and editing the example JSON files:

```bash
python tools/train_mixed.py --config configs/train_mixed.local.json --dry-run
# Or, after editing the SCoPE example:
python tools/train_mixed.py --config configs/train_scope.local.json --dry-run
python tools/evaluate.py --config configs/eval_mixed.local.json --dry-run
```

Dry runs validate paths/metadata without loading WAN or starting GPU work.
See [MIGRATION.md](docs/MIGRATION.md) before actually running these commands.
GPU model, system RAM, CUDA/driver compatibility, scheduler account/partition and
new storage roots must be confirmed for the destination cluster.

Preserve upstream licenses in the bundled source. Dataset licenses/access
agreements and collaborators' permission still apply; this archive does not
grant permission to publish HM3D, Replica, JHU or DL3DV assets or model weights.
