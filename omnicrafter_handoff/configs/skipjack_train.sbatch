#!/usr/bin/env bash
# TEMPLATE: replace account and paths; review resource COST before submitting.
#SBATCH --job-name=omnicrafter-mixed
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=omnicrafter-train-%j.log
set -euo pipefail
# 384G host RAM is an example request, NOT a measured minimum for the destination.
: "${PACKAGE_ROOT:?Set the absolute extracted package path before submission}"
: "${TRAIN_CONFIG:?Set the absolute path to the edited training JSON}"
: "${TRAIN_PYTHON:?Set the tested environment Python executable}"
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf '%s\n' 'Run with sbatch in an allocated GPU job, not on a login node.' >&2
  exit 2
fi
cd "${PACKAGE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Do not reset CUDA_VISIBLE_DEVICES and do not launch four additional srun tasks.
exec "${TRAIN_PYTHON}" -B tools/train_mixed.py --config "${TRAIN_CONFIG}"
