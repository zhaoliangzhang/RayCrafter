#!/usr/bin/env python3
"""Portable launcher for the unchanged OmniCrafter trainer. No auto-GPU selection."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common import CODE, ROOT, check_resolution, wan_assets, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--pose-manifest", type=Path, required=True)
    p.add_argument("--wan-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True, help="Published FrameCrafter initialization or a compatible OmniCrafter checkpoint")
    p.add_argument("--output", type=Path, required=True, help="New directory; checkpoints are not overwritten")
    p.add_argument("--gpus", default="0,1,2,3", help="Explicit physical CUDA GPU indices")
    p.add_argument("--port", type=int, default=29501)
    p.add_argument("--initial-step", type=int, default=0)
    p.add_argument("--steps", type=int, required=True, help="ADDITIONAL optimizer updates")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1, help="Logical sets per GPU (sequential GPU micro-forwards)")
    p.add_argument("--queries", type=int, default=3)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--nvs-ratio", type=float, default=0.5, help="Independent per-query black-image probability")
    p.add_argument("--nvs-probability", type=float, default=0.0, help="Sample-level ALL queries black; set nvs-ratio=0 when enabled")
    p.add_argument("--min-drop", type=float, default=0.0)
    p.add_argument("--max-drop", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dry-run", action="store_true", help="Validate paths and print command; no WAN/GPU loading or output writes")
    return p


def build_command(args, model_paths, tokenizer):
    check_resolution(args.height, args.width)
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or any(not i.isdigit() for i in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must be distinct comma-separated numeric indices")
    if min(args.steps, args.save_every, args.batch_size, args.queries) < 1 or args.initial_step < 0:
        raise ValueError("steps/save-every/batch-size/queries must be positive; initial-step nonnegative")
    if not (0 <= args.min_drop <= args.max_drop <= 1 and 0 <= args.nvs_ratio <= 1 and 0 <= args.nvs_probability <= 1):
        raise ValueError("Invalid dropout/task probabilities")
    if args.nvs_ratio > 0 and args.nvs_probability > 0:
        raise ValueError("Cannot combine query-level nvs-ratio and sample-level nvs-probability")
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", str(ROOT / "configs/accelerate_zero2_gpu.yaml"),
           "--num_processes", str(len(gpu_ids)), "--main_process_port", str(args.port), "--mixed_precision", "bf16",
           "-m", "model_training.train_pose_conditioned_render_queries"]
    options = {
        "dataset_base_path": args.data_root.resolve(), "dataset_metadata_path": args.train_manifest.resolve(),
        "pose_manifest_path": args.pose_manifest.resolve(), "height": args.height, "width": args.width,
        "dataset_repeat": 1, "num_dataset_samples": 0, "dataset_num_workers": 0, "max_query_slots": args.queries,
        "min_query_pixel_dropout": args.min_drop, "max_query_pixel_dropout": args.max_drop,
        "nvs_ratio": args.nvs_ratio, "nvs_probability": args.nvs_probability,
        "model_paths": json.dumps(model_paths), "tokenizer_path": tokenizer,
        "resume_checkpoint": args.init_checkpoint.resolve(), "learning_rate": args.lr, "weight_decay": 0.01,
        # Enough epochs even for tiny datasets; max_train_steps is the stopping condition.
        "num_epochs": args.steps, "train_batch_size": args.batch_size,
        "initial_global_step": args.initial_step, "max_train_steps": args.initial_step + args.steps,
        "save_steps": args.save_every, "output_path": args.output.resolve(),
        "memory_report_steps": 100, "memory_report_path": args.output.resolve() / "memory.jsonl",
        "loss_report_path": args.output.resolve() / "loss.jsonl", "remove_prefix_in_ckpt": "pipe.dit.",
        "lora_base_model": "dit", "lora_target_modules": "q,k,v,o,ffn.0,ffn.2", "lora_rank": 32,
        "extra_inputs": "conditioning_images", "new_in_dim": 420, "gradient_accumulation_steps": 1,
    }
    for key, value in options.items():
        cmd.extend([f"--{key}", str(value)])
    cmd += ["--stream_metadata", "--modify_channels", "--individual_encoding", "--initialize_model_on_cpu", "--use_gradient_checkpointing"]
    return cmd


def main():
    args = parser().parse_args()
    for path in (args.data_root, args.train_manifest, args.pose_manifest, args.init_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    model_paths, tokenizer = wan_assets(args.wan_root)
    cmd = build_command(args, model_paths, tokenizer)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "command": cmd})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = args.output / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log.open("x") as handle:
        with subprocess.Popen(cmd, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            status = proc.wait()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
