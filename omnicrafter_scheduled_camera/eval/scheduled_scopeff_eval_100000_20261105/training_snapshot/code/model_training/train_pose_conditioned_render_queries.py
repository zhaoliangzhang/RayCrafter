"""Train pose-conditioned panorama queries with aligned degraded renders."""

import accelerate

from diffsynth.core.data.pose_conditioned_render_query_dataset import (
    WanPanoramaPoseConditionedRenderQueryDataset,
)
from diffsynth.diffusion import ModelLogger, launch_training_task
from model_training.train import WanTrainingModule, wan_parser


def build_dataset(args, metadata_path, deterministic=False, max_samples=None):
    return WanPanoramaPoseConditionedRenderQueryDataset(
        base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        pose_manifest_path=args.pose_manifest_path,
        repeat=1 if deterministic else args.dataset_repeat,
        height=args.height,
        width=args.width,
        max_samples=max_samples,
        min_pixel_dropout=args.min_query_pixel_dropout,
        max_pixel_dropout=args.max_query_pixel_dropout,
        no_pixel_unshuffle=args.no_pixel_unshuffle,
        deterministic_degradation=deterministic,
        stream_metadata=args.stream_metadata,
        max_query_slots=args.max_query_slots,
        nvs_probability=args.nvs_probability,
        nvs_ratio=args.nvs_ratio,
    )


def main():
    parser = wan_parser()
    parser.description = (
        "Pose-conditioned panorama generation where every query pose also has "
        "an aligned noisy render with online random pixel dropout."
    )
    parser.add_argument("--pose_manifest_path", required=True)
    parser.add_argument("--min_query_pixel_dropout", type=float, default=0.0)
    parser.add_argument("--max_query_pixel_dropout", type=float, default=1.0)
    parser.add_argument(
        "--nvs_probability",
        type=float,
        default=0.0,
        help=(
            "Sample-level probability of replacing every query render with an "
            "exactly black NVS condition. The remaining samples use noisy "
            "renders with the configured online pixel dropout."
        ),
    )
    parser.add_argument(
        "--nvs_ratio",
        type=float,
        default=0.0,
        help=(
            "Query-level NVS ratio applied by online 2D augmentation. Each "
            "query independently receives exactly 100%% pixel dropout with "
            "this probability; otherwise its stored noisy render receives "
            "the configured cleanup dropout. Cannot be combined with "
            "--nvs_probability."
        ),
    )
    parser.add_argument(
        "--stream_metadata",
        action="store_true",
        help="Use a compact byte-offset index instead of retaining all JSONL rows in RAM.",
    )
    parser.add_argument(
        "--max_query_slots",
        type=int,
        default=None,
        help="Use only the first N contiguous query rows from each metadata group.",
    )
    parser.add_argument("--validation_metadata_path", default=None)
    parser.add_argument("--validation_report_path", default=None)
    parser.add_argument("--validation_num_samples", type=int, default=None)
    parser.add_argument("--validation_num_workers", type=int, default=0)
    parser.add_argument("--validation_seed", type=int, default=20260813)
    args = parser.parse_args()

    if not args.modify_channels or args.new_in_dim != 420:
        raise ValueError(
            "Pose-conditioned render-query training requires "
            "--modify_channels --new_in_dim 420"
        )
    if not args.individual_encoding:
        raise ValueError("Render-query training requires --individual_encoding")
    extra_inputs = {item for item in (args.extra_inputs or "").split(",") if item}
    if "conditioning_images" not in extra_inputs:
        raise ValueError(
            "Render-query training requires --extra_inputs conditioning_images"
        )

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = build_dataset(
        args,
        args.dataset_metadata_path,
        max_samples=None if args.num_dataset_samples <= 0 else args.num_dataset_samples,
    )
    validation_dataset = None
    if args.validation_metadata_path is not None:
        validation_dataset = build_dataset(
            args,
            args.validation_metadata_path,
            deterministic=True,
            max_samples=args.validation_num_samples,
        )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        modify_channels=args.modify_channels,
        new_in_dim=args.new_in_dim,
        individual_encoding=args.individual_encoding,
        resume_checkpoint=args.resume_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        validation_dataset=validation_dataset,
        args=args,
    )


if __name__ == "__main__":
    main()
