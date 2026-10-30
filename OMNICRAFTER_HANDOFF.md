# OmniCrafter handoff (for Skipjack)

`omnicrafter_handoff/` is xliu253's `OmniCrafter_cluster_handoff_20261028` package,
as updated on 2026-10-29 14:28 EDT (SCoPE training wiring), copied unmodified from
`/cis/home/xliu253/Documents/PanoLagerNVS/OmniCrafter_cluster_handoff_20261028`
on CIS. It is standalone: it does not import RayCrafter. Start with
[`omnicrafter_handoff/README.md`](omnicrafter_handoff/README.md) and
[`docs/SKIPJACK.md`](omnicrafter_handoff/docs/SKIPJACK.md).

## One file is not in git

`omnicrafter_handoff/metadata/train_cached.jsonl.gz` (the exact 200k-set training
manifest, 136,734,952 bytes) is over GitHub's 100 MB limit, so it is ignored here.
Copy it into place, then run the package's integrity check:

```bash
# from CIS, or any host that can read it
rsync -a /cis/home/xliu253/Documents/PanoLagerNVS/OmniCrafter_cluster_handoff_20261028/metadata/train_cached.jsonl.gz \
    <skipjack>:<repo>/omnicrafter_handoff/metadata/
# on Skipjack
sha256sum omnicrafter_handoff/metadata/train_cached.jsonl.gz
# f76e8b859bb4ca1c4ac924919d33f1f8c9bff8b2b12e92b93637bafabee88d09
cd omnicrafter_handoff && python tools/package_inventory.py   # "514 files ... integrity OK"
```

The check passes only with that file present and the package unedited, so run it
before making the `*.local.json` / sbatch edits the docs ask for.

The RGB training cache (~747 GB), the evaluation assets, the Wan2.1 weights and the
FrameCrafter initialization are not in the package either; `docs/MIGRATION.md` lists
what to transfer and where.

## Near-depth SCoPE run

The package trains concat or last-query-gauge SCoPE only. xliu253's newer near-depth
SCoPE experiment (`mixed_neardepth_2ref3_20261029`, running on io100) is in
[`omnicrafter_neardepth/`](omnicrafter_neardepth/README.md): its code unmodified, plus a
portable manifest builder and launcher that use this package's manifest and helpers.
