"""LoRA fine-tuning entry point for FrameCrafter.

Trains a Wan2.1-I2V-14B backbone for permutation-invariant novel view
synthesis: each frame is encoded independently, the Plucker raymap is
channel-concatenated into the DiT input, and the flow-matching SFT loss is
applied to the last N target frames of an M+N latent sequence.
"""

import argparse
import os
import warnings

import accelerate
import torch
import torch.nn as nn

from diffsynth.core.data.dataset import WanNVSDataset
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    ModelLogger,
    add_general_config,
    add_video_size_config,
    launch_training_task,
)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    """Single-DiT LoRA SFT module for FrameCrafter."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        modify_channels=False,
        new_in_dim=None,
        individual_encoding=False,
        resume_checkpoint=None,
    ):
        super().__init__()
        if not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            warnings.warn(
                "Gradient checkpointing was disabled on the CLI. The 14B DiT "
                "will OOM without it, so the training framework is forcibly "
                "enabling it."
            )
            use_gradient_checkpointing = True

        # ---- pipeline -----------------------------------------------------
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(path=tokenizer_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        self.individual_encoding = individual_encoding
        if individual_encoding:
            # Per-frame VAE encoding + per-frame y-channel paths.
            self.pipe.dit.individual_encoding = True

        if modify_channels and new_in_dim is not None:
            self._modify_model_channels(self.pipe.dit, new_in_dim, device)

        # ---- LoRA + freezing ---------------------------------------------
        effective_lora_checkpoint = resume_checkpoint if resume_checkpoint is not None else lora_checkpoint
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=effective_lora_checkpoint,
        )

        # Re-train patch_embedding from scratch: it was rebuilt with a new
        # in_dim and must remain unfrozen (LoRA only touches q/k/v/o/ffn).
        if modify_channels and new_in_dim is not None and lora_base_model is not None:
            self._unfreeze_patch_embedding(self.pipe, lora_base_model)

        # The generic LoRA loader strips non-LoRA tensors. Restore the trained
        # ray-aware patch embedding from either a released initialization or a
        # resume checkpoint after the 420-channel layer has been constructed.
        patch_checkpoint = resume_checkpoint or lora_checkpoint
        if (
            modify_channels
            and new_in_dim is not None
            and patch_checkpoint is not None
            and lora_base_model is not None
        ):
            from diffsynth.core import load_state_dict
            ckpt_sd = load_state_dict(patch_checkpoint)
            extra_state = {k: v for k, v in ckpt_sd.items() if "patch_embedding" in k}
            if extra_state:
                getattr(self.pipe, lora_base_model).load_state_dict(extra_state, strict=False)
                print(
                    f"Loaded {len(extra_state)} patch_embedding keys from "
                    f"{patch_checkpoint}"
                )
            else:
                print(f"Warning: no patch_embedding keys found in {patch_checkpoint}")
            del ckpt_sd

        # ---- training-time misc ------------------------------------------
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    # ----------------------------------------------------------------------
    # Channel surgery
    # ----------------------------------------------------------------------

    def _modify_model_channels(self, model, new_in_dim, device):
        """Replace ``patch_embedding`` so the DiT accepts ``new_in_dim``
        channels (16 latent + 4 mask + 4*100 vae context + raymap, etc.)
        instead of the stock 36. Weights of the new Conv3d are random;
        they are unfrozen for training below.
        """
        if model is None:
            return

        old_in_dim = model.in_dim
        print(f"Modifying DiT input channels: {old_in_dim} -> {new_in_dim}")

        old_pe = model.patch_embedding
        pe_device = next(old_pe.parameters()).device
        pe_dtype = next(old_pe.parameters()).dtype
        new_pe = nn.Conv3d(
            new_in_dim, model.dim,
            kernel_size=model.patch_size, stride=model.patch_size,
        ).to(device=pe_device, dtype=pe_dtype)
        model.patch_embedding = new_pe
        del old_pe

        model.in_dim = new_in_dim
        # Encode every frame independently: one temporal latent per video frame.
        model.individual_encoding = True
        print("DiT channels modified successfully")

    def _unfreeze_patch_embedding(self, pipe, lora_base_model):
        model = getattr(pipe, lora_base_model, None)
        if model is None or not hasattr(model, "patch_embedding"):
            return
        for param in model.patch_embedding.parameters():
            param.requires_grad = True
        print(f"Unfroze patch_embedding in {lora_base_model} for full training")

    # ----------------------------------------------------------------------
    # Forward / loss
    # ----------------------------------------------------------------------

    def _parse_extra_inputs(self, data, inputs_shared):
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["input_images"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def _get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_image": data["input_images"],
            "input_video": data["target_images"],
            "raymap": data.get("raymap"),
            "height": data["input_images"][0].size[1],
            "width": data["input_images"][0].size[0],
            "num_frames": len(data["target_images"]),
            "num_output_frames": len(data["target_images"]) - len(data["input_images"]),
            # Directly specify the latent temporal dimension (= M + N) for the
            # per-frame encoding path.
            "num_latent_frames": len(data["target_images"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self._parse_extra_inputs(data, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self._get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs
        return FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def wan_parser():
    parser = argparse.ArgumentParser(description="FrameCrafter LoRA training.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional override for the T5 tokenizer path.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max sampling boundary (fraction of timesteps).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min sampling boundary (fraction of timesteps).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Build the pipeline on CPU before accelerator.prepare moves it.")
    parser.add_argument("--modify_channels", default=False, action="store_true", help="Rebuild patch_embedding with --new_in_dim input channels.")
    parser.add_argument("--new_in_dim", type=int, default=None, help="New input channels for patch_embedding (required with --modify_channels).")
    parser.add_argument("--individual_encoding", default=False, action="store_true", help="Encode every frame independently (1 temporal latent per frame).")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a previously saved checkpoint (.safetensors). Restores both LoRA and patch_embedding weights.")
    parser.add_argument("--sampling_strategy", type=str, default="prob_random",
                        choices=["all_random", "prob_random", "all_window", "curriculum"],
                        help="Temporal window length policy for the dataset.")
    parser.add_argument("--num_dataset_samples", type=int, default=1000, help="Use the first N scenes (sorted) from dataset_base_path.")
    parser.add_argument("--no_pixel_unshuffle", default=False, action="store_true",
                        help="Use bilinear downsampling instead of PixelUnshuffle for the raymap.")
    parser.add_argument("--num_input_frames", type=int, default=None,
                        help="M (context) frames. Pair with --num_output_frames for fixed M-to-N. Omit both for random M-to-N.")
    parser.add_argument("--num_output_frames", type=int, default=None, help="N (target) frames; defaults to 1 when M is fixed.")
    parser.add_argument("--min_input_frames", type=int, default=3, help="Minimum M in random M-to-N mode.")
    parser.add_argument("--min_output_frames", type=int, default=1, help="Minimum N in random M-to-N mode.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if args.resolution_schedule:
        resolutions = args.resolution_schedule.split(",")
        parsed = []
        for item in resolutions:
            height, width = item.lower().split("x", 1)
            height, width = int(height), int(width)
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(f"Scheduled resolution must be divisible by 16: {item}")
            parsed.append((height, width))
        args.height, args.width = parsed[0]
        args.num_epochs = len(parsed)
        if args.max_train_steps is None:
            args.max_train_steps = len(parsed)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = WanNVSDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        height_division_factor=8,
        width_division_factor=8,
        time_division_factor=4,
        time_division_remainder=1,
        sampling_strategy=args.sampling_strategy,
        num_dataset_samples=args.num_dataset_samples,
        no_pixel_unshuffle=args.no_pixel_unshuffle,
        num_input_frames=args.num_input_frames,
        num_output_frames=args.num_output_frames,
        min_input_frames=args.min_input_frames,
        min_output_frames=args.min_output_frames,
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
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
