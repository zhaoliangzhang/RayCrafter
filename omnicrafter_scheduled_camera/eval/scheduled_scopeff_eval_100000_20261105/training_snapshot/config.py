"""Two independent continuations, sharing only immutable code and input data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ.get('CAMERA_VARIANT', 'scope_ff')
if VARIANT not in ('scope_ff', 'unirope'):
    raise ValueError('CAMERA_VARIANT must be scope_ff or unirope')
HOST = 'io104' if VARIANT == 'scope_ff' else 'io105'
BASE_STEP = 50000
ADDITIONAL_STEPS = 300000
FINAL_STEP = BASE_STEP + ADDITIONAL_STEPS
RAMP_STEPS = 10000
BASE = Path('/cis/net/io81/data/xliu253/omniraycrafter')
INITIALIZATION = BASE / 'mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/step-50000.safetensors'
SOURCE = BASE / 'mixed_rayrope_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/data_manifest'
OUT = BASE / f'mixed_concat_scheduled_{VARIANT}_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_{HOST}'
LOCAL = Path(f'/dev/shm/xliu253_scheduled_{VARIANT}_20261104')
PYTHON = '/cis/home/xliu253/micromamba/envs/framecrafter/bin/python'
GPU_UUIDS = {
 'io104': ['GPU-de2572d0-76cb-919a-1321-abe91e3a44a2', 'GPU-0b1d0897-aec2-700e-7f6e-26575c1afb41',
           'GPU-168a3e36-1f09-c2dc-a903-a3b44c4c1399', 'GPU-e9e3ef51-e821-78af-1246-cfd8821523b7'],
 'io105': ['GPU-314252fd-a4aa-427f-3904-33f7505866b2', 'GPU-20b7af2c-60c3-4d8e-e663-7b010613915d',
           'GPU-f98a7919-0878-086c-e3da-080c5d4c17da', 'GPU-738b5b04-56d0-88e2-adbb-8f9e9620a0c6'],
}[HOST]

def injection_weight(completed_step, ramp_steps=RAMP_STEPS):
    if ramp_steps <= 0 or completed_step < BASE_STEP:
        raise ValueError('Invalid injection schedule')
    return min((completed_step - BASE_STEP) / ramp_steps, 1.)
