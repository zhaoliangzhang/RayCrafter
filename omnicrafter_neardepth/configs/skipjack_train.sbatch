#!/usr/bin/env bash
# TEMPLATE: replace account, partition and paths; review resource COST before submitting.
# Resubmitting the same job continues from the newest step checkpoint in the run's output.
#SBATCH --job-name=omnicrafter-neardepth
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=omnicrafter-neardepth-%j.log
set -euo pipefail
# 384G host RAM is an example request, not a measured minimum (io100 ranks load the DiT on CPU).
: "${REPO_ROOT:?Set the absolute RayCrafter checkout path (branch omnicrafter-skipjack)}"
: "${TRAIN_CONFIG:?Set the absolute path to the edited training JSON}"
: "${TRAIN_PYTHON:?Set the tested environment Python executable}"
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf '%s\n' 'Run with sbatch in an allocated GPU job, not on a login node.' >&2
  exit 2
fi
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Do not reset CUDA_VISIBLE_DEVICES and do not launch four srun tasks: accelerate starts the 4 ranks.
exec "${TRAIN_PYTHON}" -B omnicrafter_neardepth/tools/train_neardepth.py --config "${TRAIN_CONFIG}"
