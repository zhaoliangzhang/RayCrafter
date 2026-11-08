import json
import hashlib
import os
import resource
import time

import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    validation_dataset: torch.utils.data.Dataset = None,
    args=None,
):
    """Flow-matching SFT training loop with optional W&B logging."""
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    train_batch_size = int(
        getattr(args, "train_batch_size", 1) if args is not None else 1
    )
    if train_batch_size < 1:
        raise ValueError(
            f"train_batch_size must be positive, got {train_batch_size}"
        )

    initial_global_step = int(
        getattr(args, "initial_global_step", 0) if args is not None else 0
    )
    if initial_global_step < 0:
        raise ValueError(
            f"initial_global_step must be non-negative, got {initial_global_step}"
        )
    if (
        args is not None
        and args.max_train_steps is not None
        and args.max_train_steps <= initial_global_step
    ):
        raise ValueError(
            "max_train_steps must be greater than initial_global_step; "
            f"got {args.max_train_steps} and {initial_global_step}"
        )
    # ModelLogger owns checkpoint numbering. Keep it aligned with the reported
    # cumulative step so continuation checkpoints cannot overwrite earlier ones.
    model_logger.num_steps = initial_global_step

    use_wandb = args is not None and getattr(args, "wandb_project", None) is not None
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        # The diffusion pipeline consumes one panorama set at a time. Preserve
        # the historical dict interface for batch size one, and expose a list
        # for larger logical batches so their independent losses can be
        # averaged into one optimizer update below.
        collate_fn=lambda samples: samples[0] if len(samples) == 1 else samples,
        num_workers=num_workers,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=getattr(args, "validation_num_workers", 0),
        )

    if validation_dataloader is None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    else:
        model, optimizer, dataloader, scheduler, validation_dataloader = accelerator.prepare(
            model, optimizer, dataloader, scheduler, validation_dataloader
        )

    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    def report_memory(step, loss=None, epoch_id=None):
        if accelerator.device.type != "cuda":
            return
        torch.cuda.synchronize(accelerator.device)
        local = torch.tensor(
            [
                accelerator.process_index,
                torch.cuda.memory_allocated(accelerator.device),
                torch.cuda.max_memory_allocated(accelerator.device),
                torch.cuda.memory_reserved(accelerator.device),
                torch.cuda.max_memory_reserved(accelerator.device),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        ).unsqueeze(0)
        gathered = accelerator.gather(local).cpu().tolist()
        if not accelerator.is_main_process:
            return
        records = []
        for values in gathered:
            records.append(
                {
                    "step": step,
                    "epoch": epoch_id,
                    "height": int(getattr(dataset, "height", 0)),
                    "width": int(getattr(dataset, "width", 0)),
                    "train_batch_size_per_gpu": train_batch_size,
                    "global_batch_size": (
                        train_batch_size
                        * accelerator.num_processes
                        * accelerator.gradient_accumulation_steps
                    ),
                    "rank": int(values[0]),
                    "cuda_allocated_bytes": int(values[1]),
                    "cuda_peak_allocated_bytes": int(values[2]),
                    "cuda_reserved_bytes": int(values[3]),
                    "cuda_peak_reserved_bytes": int(values[4]),
                    "process_peak_rss_bytes": int(values[5]),
                    "loss": None if loss is None else float(loss),
                }
            )
        print("MEMORY_REPORT " + json.dumps(records, sort_keys=True))
        report_path = getattr(args, "memory_report_path", None) if args is not None else None
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "a") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    def run_validation(step):
        if validation_dataloader is None:
            return None
        accelerator.wait_for_everyone()
        started = time.monotonic()
        module_training_states = {
            module: module.training for module in model.modules()
        }
        model.eval()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if accelerator.device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(accelerator.device)

        local_loss_sum = 0.0
        local_count = 0
        local_cleanup_targets = 0
        local_nvs_targets = 0
        validation_seed = getattr(args, "validation_seed", 20260813)
        try:
            with torch.no_grad():
                for data in tqdm(
                    validation_dataloader,
                    desc=f"validation step {step}",
                    disable=not accelerator.is_main_process,
                ):
                    sample_id = str(data.get("sample_id", local_count))
                    digest = hashlib.sha256(sample_id.encode()).digest()
                    sample_seed = (
                        validation_seed + int.from_bytes(digest[:8], "little")
                    ) % (2**31)
                    torch.manual_seed(sample_seed)
                    if accelerator.device.type == "cuda":
                        torch.cuda.manual_seed(sample_seed)
                    loss = model(data)
                    if not torch.isfinite(loss).all():
                        raise RuntimeError(
                            f"Non-finite validation loss for {sample_id} at step {step}"
                        )
                    local_loss_sum += float(loss.detach().double().item())
                    local_count += 1
                    tasks = data.get("target_task_types", [])
                    local_cleanup_targets += tasks.count("cleanup")
                    local_nvs_targets += tasks.count("nvs")
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, accelerator.device)
            for module, training in module_training_states.items():
                module.training = training

        stats = torch.tensor(
            [
                local_loss_sum,
                local_count,
                local_cleanup_targets,
                local_nvs_targets,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        stats = accelerator.reduce(stats, reduction="sum")
        elapsed = torch.tensor(
            [time.monotonic() - started],
            dtype=torch.float64,
            device=accelerator.device,
        )
        elapsed = accelerator.gather(elapsed).max().item()
        if not accelerator.is_main_process:
            return None

        record = {
            "step": step,
            "checkpoint": os.path.join(
                model_logger.output_path, f"step-{step}.safetensors"
            ),
            "samples": int(stats[1].item()),
            "cleanup_targets": int(stats[2].item()),
            "nvs_targets": int(stats[3].item()),
            "flow_matching_loss": float(stats[0].item() / stats[1].item()),
            "elapsed_seconds": float(elapsed),
            "height": int(getattr(validation_dataset, "height", 0)),
            "width": int(getattr(validation_dataset, "width", 0)),
        }
        print("VALIDATION_REPORT " + json.dumps(record, sort_keys=True))
        report_path = getattr(args, "validation_report_path", None)
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if use_wandb:
            import wandb
            wandb.log(
                {"val/flow_matching_loss": record["flow_matching_loss"]},
                step=step,
            )
        return record

    global_step = initial_global_step
    last_validation_step = initial_global_step
    # The dataset uses these to drive curriculum sampling.
    if hasattr(dataset, "num_epochs"):
        dataset.num_epochs = num_epochs
    for epoch_id in range(num_epochs):
        resolution_schedule = getattr(args, "resolution_schedule", None) if args is not None else None
        if resolution_schedule:
            resolutions = []
            for item in resolution_schedule.split(","):
                height, width = item.lower().split("x", 1)
                resolutions.append((int(height), int(width)))
            dataset.height, dataset.width = resolutions[epoch_id]
            if accelerator.is_main_process:
                print(
                    f"RESOLUTION_PROBE epoch={epoch_id} "
                    f"resolution={dataset.height}x{dataset.width}"
                )
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(accelerator.device)
        if hasattr(dataset, "current_epoch"):
            dataset.current_epoch = epoch_id
        for data in tqdm(dataloader):
            step_started = time.monotonic()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                logical_batch = data if isinstance(data, list) else [data]
                detached_losses = []
                for sample in logical_batch:
                    sample_loss = model(sample)
                    if not torch.isfinite(sample_loss).all():
                        raise RuntimeError(
                            "Non-finite training loss at step "
                            f"{global_step + 1}: {sample_loss}"
                        )
                    accelerator.backward(sample_loss / len(logical_batch))
                    detached_losses.append(sample_loss.detach())
                loss = torch.stack(detached_losses).mean()
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                global_step += 1
                memory_report_steps = max(
                    1, int(getattr(args, "memory_report_steps", 1))
                )
                if global_step % memory_report_steps == 0:
                    report_memory(global_step, loss.detach().float().item(), epoch_id)

                loss_report_path = (
                    getattr(args, "loss_report_path", None) if args is not None else None
                )
                if loss_report_path and accelerator.is_main_process:
                    os.makedirs(os.path.dirname(os.path.abspath(loss_report_path)), exist_ok=True)
                    loss_record = {
                        "step": global_step,
                        "epoch": epoch_id,
                        "loss": float(loss.detach().float().item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "step_seconds": float(time.monotonic() - step_started),
                    }
                    with open(loss_report_path, "a") as handle:
                        handle.write(json.dumps(loss_record, sort_keys=True) + "\n")

                if use_wandb and accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch_id,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)

            if (
                validation_dataloader is not None
                and save_steps is not None
                and global_step % save_steps == 0
            ):
                run_validation(global_step)
                last_validation_step = global_step

            if args is not None and args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if args is not None and args.max_train_steps is not None and global_step >= args.max_train_steps:
            break
    if args is None or not getattr(args, "skip_final_checkpoint", False):
        model_logger.on_training_end(accelerator, model, save_steps)
    if validation_dataloader is not None and global_step != last_validation_step:
        run_validation(global_step)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()
