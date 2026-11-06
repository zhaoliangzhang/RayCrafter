#!/usr/bin/env bash
# Concat + scheduled SCoPE or RayRoPE continuation, 50k -> 100k, on 4xH100.
# Submit from the root of the worktree that holds this branch:
#   CAMERA_VARIANT=scope   sbatch -J oc-sched-scope   omnicrafter_scheduled_camera/train_4gpu.sbatch
#   CAMERA_VARIANT=rayrope sbatch -J oc-sched-rayrope omnicrafter_scheduled_camera/train_4gpu.sbatch
# The run saves its full state (resume/) every 2000 steps and 30 min before the walltime.
# Resubmitting the same command continues from it; once complete, a resubmit only reports that.
#SBATCH --partition=h100
#SBATCH --account=rchella4_videolincs
#SBATCH --qos=jhu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=32:00:00
#SBATCH --output=/weka/home/jhu/dli90/RayCrafter/bash_outputs/%x-%j.out
#SBATCH --error=/weka/home/jhu/dli90/RayCrafter/bash_outputs/%x-%j.err
set -euo pipefail

: "${CAMERA_VARIANT:?set CAMERA_VARIANT to scope or rayrope}"
TIME_LIMIT_HOURS=${TIME_LIMIT_HOURS:-31.5}
export JOB_START_EPOCH=$(date +%s)
cd "${SLURM_SUBMIT_DIR:?}/omnicrafter_scheduled_camera"

ENV_PREFIX=${ENV_PREFIX:-/weka/projects/rchella4/dli90/envs/omnicrafter}
export PATH="$ENV_PREFIX/bin:$PATH"
export VIRTUAL_ENV="$ENV_PREFIX"
# As xliu253's run_job.py, with node-local compile caches.
export CAMERA_VARIANT PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
CACHE="${TMPDIR:-/tmp}/${USER}/oc_scheduled_${CAMERA_VARIANT}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
unset TORCH_COMPILE_DISABLE

eval "$(python -B -c 'import config as c; print(f"OUT={c.OUT} INIT={c.INITIALIZATION} SOURCE={c.SOURCE} WAN={c.WAN} PATH_MAP={c.PATH_MAP} FINAL_STEP={c.FINAL_STEP} RAMP_STEPS={c.RAMP_STEPS}")')"
for path in "$INIT" "$SOURCE/train.jsonl" "$WAN/Wan-AI/Wan2.1-I2V-14B-480P" \
            /weka/projects/rchella4/zzhan288/data/dl3dv_points/train/priors/samples \
            /weka/projects/rchella4/zzhan288/data/framecrafter_5input/HM3D; do
  for _ in $(seq 60); do [[ -e "$path" ]] && break; sleep 5; done
  [[ -e "$path" ]] || { echo "missing $path" >&2; exit 1; }
done
mkdir -p "$OUT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
echo "host $(hostname)  job ${SLURM_JOB_ID}  variant ${CAMERA_VARIANT}  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  output ${OUT}"
nvidia-smi -L

TORCHRUN=(python -B -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4)
CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 python -B -u preflight.py 2>&1 | tee "$OUT/logs/preflight_${STAMP}.log"
if [[ ! -f "$OUT/resume/meta.json" && ! -f "$OUT/resume.old/meta.json" ]]; then
  "${TORCHRUN[@]}" gpu_smoke.py 2>&1 | tee "$OUT/logs/gpu_correctness_${STAMP}.log"
fi
"${TORCHRUN[@]}" train.py --wan "$WAN" --initialization-checkpoint "$INIT" \
  --manifest "$SOURCE/train.jsonl" --offsets "$SOURCE/train.offsets.npy" --output "$OUT" \
  --max-steps "$FINAL_STEP" --save-every 2000 --ramp-steps "$RAMP_STEPS" --nvs-probability 0.5 \
  --path-map "$PATH_MAP" --time-limit-hours "$TIME_LIMIT_HOURS" 2>&1 | tee "$OUT/logs/train_${STAMP}.log"
