#!/usr/bin/env bash
# OmniCrafter backbone from Wan weights + channel concat on 4 GPUs on CIS (omnicrafter_concat_wan/train.py).
# Run on the GPU node, from anywhere:
#   bash omnicrafter_concat_wan/train_4gpu.sh
#   CONFIG=<other yaml> NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash omnicrafter_concat_wan/train_4gpu.sh   # e.g. a smoke test
# Rerunning continues from <output_dir>/resume (full state; same GPU count). After max_steps the final
# full state is kept in <output_dir>/resume_step<max_steps>; a rerun then only reports completion.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG=${CONFIG:-omnicrafter_concat_wan/configs/omnicrafter_mix_concat_wan_4gpu_50k.yaml}
NPROC=${NPROC:-4}
PORT=${PORT:-29651}
ENV_PREFIX=${ENV_PREFIX:-/cis/home/dli90/miniconda3/envs/omnicrafter}
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="/tmp/${USER}/triton_omnicrafter_concat_wan"

echo "host $(hostname)  config ${CONFIG}  processes ${NPROC}  commit $(git rev-parse --short HEAD)"
nvidia-smi -L
python -c "import torch, accelerate, deepspeed; print('torch', torch.__version__, 'accelerate', accelerate.__version__, 'deepspeed', deepspeed.__version__)"

exec python -m accelerate.commands.launch \
  --config_file omnicrafter_scopeff/accelerate_zero2.yaml \
  --num_processes "$NPROC" --mixed_precision bf16 \
  --main_process_port "$PORT" \
  omnicrafter_concat_wan/train.py --config "$CONFIG"
