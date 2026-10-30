#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/mixed_neardepth_2ref3_20261029
[[ "$(hostname -s)" == io100 ]] || { echo 'Run on io100 only'; exit 1; }
RUN_OUTPUT=/cis/net/io81/data/xliu253/omniraycrafter/mixed_scope_neardepth_q3_dl3dv100k_hm3d60k_jhu40k_b1_100k_io100
RUN_PYTHON=/cis/home/xliu253/micromamba/envs/framecrafter/bin/python
mkdir -p "${RUN_OUTPUT}/logs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec > >(tee -a "${RUN_OUTPUT}/logs/queue_$(date +%Y%m%d_%H%M%S).log") 2>&1
date -Is
"${RUN_PYTHON}" -B -u prepare_cached_manifest.py
"${RUN_PYTHON}" -B -u configure.py --architecture scope --initialization published
"${RUN_PYTHON}" -B -u stage_wan.py
"${RUN_PYTHON}" -B -u queue_io100.py
