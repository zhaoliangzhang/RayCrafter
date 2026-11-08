"""Runtime-only migration of the existing SCoPE-FF 100k evaluation to io95."""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'scheduled_scopeff_eval_100000_20261105'
sys.path.insert(0, str(ORIGINAL))
import eval_shared as original

OUT = original.OUT
RUNTIME = OUT / 'runtime/resume_io95_20261106'
CACHE_ROOT = Path('/export/io95/data/xliu253/scheduled_scopeff100k_resume_20261106')
WAN = Path('/export/io95/data/xliu253/framecrafter_wan21')
DREAM_CACHE = Path('/export/io95/data/xliu253/concat_scopeff_unirope_eval_22000_20261103/dreamsim')
FID_SOURCE = Path('/export/io95/data/xliu253/concat_scopeff_unirope_eval_22000_20261103/fid/inception-2015-12-05.pt')
GPUS = [
    'GPU-fcfa9a32-08ec-b6bc-514c-e94d248b4f19',
    'GPU-8b4c39b3-d294-acf1-a3e0-f4e3d45549ea',
    'GPU-d01abde5-f58a-98df-e2af-b27f390a6666',
    'GPU-d97089ea-d914-5b89-ac07-12acaee2e2fb',
    'GPU-57a353c7-a517-307d-22a7-44ab0ed9e876',
    'GPU-78f722f1-1433-ead7-4912-fe329a0c654a',
    'GPU-6972292e-925b-5de2-83c4-e5b6d2e028fe',
    'GPU-b75ff6d9-e49d-a797-cd09-ee59de580966',
]
NUM_SHARDS = 8
VRAM_LIMIT_GIB = 32


def adapter_hashes():
    return {str(p): original.sha256(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py', '.sh', '.json')}


def configure():
    for name, value in dict(HOST='io95', WAN=WAN, CACHE_ROOT=CACHE_ROOT,
                            CACHE=CACHE_ROOT, DREAM_SOURCE=DREAM_CACHE,
                            DREAM_CACHE=DREAM_CACHE, FID_SOURCE=FID_SOURCE,
                            GPUS=GPUS, GPU_INDICES=list(range(8)),
                            NUM_SHARDS=NUM_SHARDS, VRAM_LIMIT_GIB=VRAM_LIMIT_GIB).items():
        setattr(original, name, value)
    # Do not modify original source hashes, checkpoint, manifest, output,
    # seeds, resolutions, camera math, or sampling configuration.
    return original
