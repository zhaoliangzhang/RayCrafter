"""Two independent continuations on Skipjack, sharing only immutable code and input data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ.get('CAMERA_VARIANT')
if VARIANT not in ('scope', 'rayrope'):
    raise ValueError('Set CAMERA_VARIANT to scope or rayrope')
BASE_STEP = 50000
ADDITIONAL_STEPS = 50000
FINAL_STEP = BASE_STEP + ADDITIONAL_STEPS
RAMP_STEPS = 10000
BASE = Path('/weka/projects/rchella4/dli90/omnicrafter_outputs')
# Copies of xliu253's concat step-50000 adapter and of their 200k-set data_manifest/.
INITIALIZATION = BASE / 'mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/step-50000.safetensors'
SOURCE = BASE / 'data_manifest'
OUT = BASE / f'mixed_concat_scheduled_{VARIANT}_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus50k_skipjack'
WAN = Path('/weka/projects/rchella4/zzhan288/FrameCrafter/models')
# The manifest keeps xliu253's io104 image paths; this maps them to the same files on weka.
PATH_MAP = ROOT / 'image_path_map.skipjack.json'

def injection_weight(completed_step, ramp_steps=RAMP_STEPS):
    if ramp_steps <= 0 or completed_step < BASE_STEP:
        raise ValueError('Invalid injection schedule')
    return min((completed_step - BASE_STEP) / ramp_steps, 1.)
