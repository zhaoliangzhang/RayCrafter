#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/mixed_projection_2ref3_20261023
OUT=/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104
mkdir -p "${OUT}/logs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -u queue_io104.py 2>&1 | tee -a "${OUT}/logs/queue_$(date +%Y%m%d_%H%M%S).log"
