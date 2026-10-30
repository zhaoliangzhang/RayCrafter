#!/usr/bin/env bash
# TEMPLATE ONLY: set account, partition, RAM, walltime and environment for your cluster.
#SBATCH --job-name=omnicrafter-mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=omnicrafter-%j.log
set -euo pipefail
: "${PACKAGE_ROOT:?Set PACKAGE_ROOT to the extracted package directory}"
: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to your edited absolute JSON config path}"
# Activate your tested environment here. Do not override scheduler CUDA_VISIBLE_DEVICES.
cd "${PACKAGE_ROOT}"
python tools/train_mixed.py --config "${TRAIN_CONFIG}"
