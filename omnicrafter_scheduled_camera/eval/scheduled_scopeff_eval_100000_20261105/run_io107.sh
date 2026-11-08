#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/scheduled_scopeff_eval_100000_20261105
export CAMERA_VARIANT=scope_ff PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
out=/cis/net/io81/data/xliu253/omniraycrafter/mixed_concat_scheduled_scope_ff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io104/evaluations/step100000_hm3d_replica_jhu_dl3dv_q3_20261105
mkdir -p "$out/logs"
log="$out/logs/controller_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1
exec /cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -u run.py
