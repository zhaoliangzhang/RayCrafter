"""Concat + SCoPE-FF continuation on Skipjack, without an injection schedule."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ.get('CAMERA_VARIANT')
if VARIANT != 'scope_ff':
    raise ValueError('This branch trains concat + SCoPE-FF only: set CAMERA_VARIANT=scope_ff')
BASE_STEP = 50000
ADDITIONAL_STEPS = 50000
FINAL_STEP = BASE_STEP + ADDITIONAL_STEPS
# No schedule: lambda = 1 from the first update. SCoPE-FF's zero-initialized
# output layer still makes the model before that update the concat model.
RAMP_STEPS = 0
BASE = Path('/weka/projects/rchella4/dli90/omnicrafter_outputs')
# Copies of xliu253's concat step-50000 adapter and of their 200k-set data_manifest/.
INITIALIZATION = BASE / 'mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/step-50000.safetensors'
SOURCE = BASE / 'data_manifest'
OUT = BASE / 'mixed_concat_scopeff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus50k_skipjack'
WAN = Path('/weka/projects/rchella4/zzhan288/FrameCrafter/models')
# The manifest keeps xliu253's io104 image paths; this maps them to the same files on weka.
PATH_MAP = ROOT / 'image_path_map.skipjack.json'

def injection_weight(completed_step, ramp_steps=RAMP_STEPS):
    """0 before the first update; then a linear ramp over ``ramp_steps``, or 1 when it is 0."""
    if ramp_steps < 0 or completed_step < BASE_STEP:
        raise ValueError('Invalid injection schedule')
    if ramp_steps == 0:
        return 0. if completed_step == BASE_STEP else 1.
    return min((completed_step - BASE_STEP) / ramp_steps, 1.)

def schedule_type(ramp_steps=RAMP_STEPS):
    return 'linear' if ramp_steps else 'constant'
