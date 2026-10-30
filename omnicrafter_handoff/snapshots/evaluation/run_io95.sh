#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/mixed_projection_eval_50000_20261028
EVAL_OUT=/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/evaluations/step50000_dl3dv_hm3d_replica_jhu_q3_20261028
mkdir -p "${EVAL_OUT}/logs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -u run.py \
  2>&1 | tee -a "${EVAL_OUT}/logs/controller_$(date +%Y%m%d_%H%M%S).log"
