#!/usr/bin/env bash
set -euo pipefail
cd /cis/home/xliu253/Documents/PanoLagerNVS/mixed_projection_2ref3_20261023
OUT=/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104
mkdir -p "${OUT}/logs"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/cis/home/xliu253/micromamba/envs/framecrafter/bin/python -B -u prepare_manifest.py --workers 32 2>&1 | tee "${OUT}/logs/prepare_$(date +%Y%m%d_%H%M%S).log"
