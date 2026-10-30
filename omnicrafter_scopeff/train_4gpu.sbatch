#!/usr/bin/env bash
# OmniCrafter + scope-ff on 4xH100 (omnicrafter_scopeff/train.py).
#   sbatch omnicrafter_scopeff/train_4gpu.sbatch
#   CONFIG=<other yaml> sbatch -t 01:30:00 omnicrafter_scopeff/train_4gpu.sbatch   # e.g. a smoke test
#   Fewer / other GPUs (smoke tests only; NPROC must match --gres):
#   CONFIG=<yaml> NPROC=2 sbatch -p a100 --gres=gpu:a100:2 --cpus-per-task=32 --mem=192G -t 01:30:00 omnicrafter_scopeff/train_4gpu.sbatch
# Resubmitting continues from <output_dir>/resume (full state; 4 GPUs). After max_steps the final
# full state is kept in <output_dir>/resume_step<max_steps>; a resubmit then only reports completion.
#SBATCH --job-name=oc-scopeff-4g
#SBATCH --partition=h100
#SBATCH --account=rchella4_wriva
#SBATCH --qos=jhu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=72:00:00
#SBATCH --chdir=/weka/home/jhu/dli90/RayCrafter
#SBATCH --output=bash_outputs/%x-%j.out
#SBATCH --error=bash_outputs/%x-%j.err
set -euo pipefail

CONFIG=${CONFIG:-omnicrafter_scopeff/configs/omnicrafter_mix_scope_ff_4gpu_50k.yaml}
NPROC=${NPROC:-4}
ENV_PREFIX=${ENV_PREFIX:-/weka/projects/rchella4/dli90/envs/omnicrafter}
export PATH="$ENV_PREFIX/bin:$PATH"
export VIRTUAL_ENV="$ENV_PREFIX"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/${USER}/triton_omnicrafter_scopeff"

# Wait for the data and weight mounts.
for path in /weka/projects/rchella4/zzhan288/data/dl3dv_points/train/priors/samples \
            /weka/projects/rchella4/zzhan288/data/framecrafter_5input/near_depth.json \
            /weka/projects/rchella4/zzhan288/FrameCrafter/models/Wan-AI/Wan2.1-I2V-14B-480P; do
  for _ in $(seq 60); do [[ -e "$path" ]] && break; sleep 5; done
  [[ -e "$path" ]] || { echo "missing $path" >&2; exit 1; }
done

echo "host $(hostname)  job ${SLURM_JOB_ID}  config ${CONFIG}  processes ${NPROC}"
nvidia-smi -L
python -c "import torch, accelerate, deepspeed; print('torch', torch.__version__, 'accelerate', accelerate.__version__, 'deepspeed', deepspeed.__version__)"

# Launch in the batch step, as omnicrafter_handoff/configs/skipjack_train.sbatch does (no srun).
python -m accelerate.commands.launch \
  --config_file omnicrafter_scopeff/accelerate_zero2.yaml \
  --num_processes "$NPROC" --mixed_precision bf16 \
  --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
  omnicrafter_scopeff/train.py --config "$CONFIG"
