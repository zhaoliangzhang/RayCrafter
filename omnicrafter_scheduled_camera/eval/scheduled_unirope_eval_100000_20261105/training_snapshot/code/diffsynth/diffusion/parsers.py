import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images.")
    parser.add_argument("--width", type=int, default=None, help="Width of images.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video (M + N).")
    parser.add_argument(
        "--resolution_schedule",
        type=str,
        default=None,
        help="Optional comma-separated HxW schedule, one resolution per epoch.",
    )
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g. 'Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors'. Comma-separated for multiple.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated (e.g. 'input_image').")
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help=(
            "Logical samples per GPU and optimizer step. Samples are evaluated "
            "independently and their losses are averaged before the update."
        ),
    )
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to fully train (no LoRA), comma-separated.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional global optimizer-step limit, useful for smoke tests and memory probes.",
    )
    parser.add_argument(
        "--initial_global_step",
        type=int,
        default=0,
        help=(
            "Cumulative optimizer-step offset for weight-only continuation. "
            "Checkpoint names and reports continue from this value."
        ),
    )
    parser.add_argument(
        "--memory_report_path",
        type=str,
        default=None,
        help="Optional JSONL output for per-rank CUDA and process-memory peaks.",
    )
    parser.add_argument(
        "--memory_report_steps",
        type=int,
        default=1,
        help="Collect synchronized memory statistics every N steps.",
    )
    parser.add_argument(
        "--loss_report_path",
        type=str,
        default=None,
        help="Optional JSONL output for the main-rank training loss at every step.",
    )
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Step interval for intermediate checkpoints. If None, checkpoints are saved per-epoch.")
    parser.add_argument(
        "--skip_final_checkpoint",
        default=False,
        action="store_true",
        help="Do not save a final checkpoint, useful for short memory-capacity probes.",
    )
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which submodule LoRA is added to (e.g. 'dit').")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Comma-separated LoRA target module names.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to a LoRA checkpoint to initialise from.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser


def add_wandb_config(parser: argparse.ArgumentParser):
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name. If set, enables wandb logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_wandb_config(parser)
    return parser
