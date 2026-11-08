#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/scheduled_scopeff100k_resume_io95_20261106
test "$(hostname -s)" = io95
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CAMERA_VARIANT=scope_ff
exec /cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -u run.py
