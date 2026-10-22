"""Rolling full-training checkpoint.

``transformer_step<N>.safetensors`` stays a bf16 weight snapshot for eval.
``output_dir/resume/`` is the one checkpoint a restarted job continues from:
fp32 master weights plus AdamW (moments and step). It is overwritten every
``resume_interval`` steps, and again on a clean stop.

ZeRO-1 keeps each rank's optimizer shard on that rank. Tensors are streamed
to disk one at a time so the save never gathers the full AdamW state onto
the host (the training jobs only have a few tens of GB of RAM).
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Optimizer

_DTYPE_TO_STR = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_VERSION = 1


def save_training_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    directory: Path,
    rank: int,
    world_size: int,
) -> None:
    """Overwrite ``directory`` with the full training state at ``step``."""
    directory = Path(directory)
    partial = _partial_dir(directory)
    backup = _prev_dir(directory)
    _barrier(world_size)
    if rank == 0:
        # A finished partial is the newest complete checkpoint (the swap into
        # ``directory`` did not happen). Promote it before replacing it.
        finalize_resume_dir(directory)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
    _barrier(world_size)

    if rank == 0:
        _stream_save(partial / "model.safetensors", module.state_dict().items())

    local = _local_optimizer(optimizer)
    sharded = isinstance(optimizer, ZeroRedundancyOptimizer)
    names = _local_param_names(module, local)
    optim_name = f"optim_rank{rank}.safetensors" if sharded else "optim.safetensors"
    if sharded or rank == 0:
        _stream_save(
            partial / optim_name,
            _optimizer_tensors(local),
            metadata={"param_names": json.dumps(names)},
        )
    _barrier(world_size)

    if rank == 0:
        meta = {
            "version": _VERSION,
            "step": int(step),
            "world_size": int(world_size),
            "shard_optimizer": bool(sharded),
        }
        meta_path = partial / "meta.json"
        with open(meta_path, "w") as meta_file:
            meta_file.write(json.dumps(meta))
            meta_file.flush()
            os.fsync(meta_file.fileno())
        if backup.exists():
            shutil.rmtree(backup)
        if directory.exists():
            directory.rename(backup)
        partial.rename(directory)
        if backup.exists():
            shutil.rmtree(backup)
        print(f"resume checkpoint → {directory} (step {step})", flush=True)
    _barrier(world_size)


def load_training_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: Path,
    rank: int,
    world_size: int,
) -> int:
    """Load weights and AdamW state. Returns the step that checkpoint finished."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if int(meta.get("version", 0)) != _VERSION:
        raise RuntimeError(f"unsupported resume checkpoint version in {directory}: {meta.get('version')}")
    sharded = bool(meta["shard_optimizer"])
    if int(meta["world_size"]) != int(world_size) or sharded != isinstance(optimizer, ZeroRedundancyOptimizer):
        raise RuntimeError(
            f"resume checkpoint {directory} was saved with world_size={meta['world_size']} "
            f"shard_optimizer={sharded}, this job is world_size={world_size} "
            f"shard_optimizer={isinstance(optimizer, ZeroRedundancyOptimizer)}. "
            "Resume needs the same number of GPUs and the same shard_optimizer setting."
        )

    _load_module_(module, directory / "model.safetensors")
    local = _local_optimizer(optimizer)
    optim_name = f"optim_rank{rank}.safetensors" if sharded else "optim.safetensors"
    _load_optimizer_(local, directory / optim_name, _local_param_names(module, local))
    _barrier(world_size)
    return int(meta["step"])


def validate_resume_meta(meta: dict, world_size: int, shard_optimizer: bool) -> None:
    """Raise before the model loads if this job cannot continue the checkpoint."""
    if int(meta.get("version", 0)) != _VERSION:
        raise RuntimeError(f"unsupported resume checkpoint version: {meta.get('version')}")
    if int(meta["world_size"]) != int(world_size) or bool(meta["shard_optimizer"]) != bool(shard_optimizer):
        raise RuntimeError(
            f"resume checkpoint was saved with world_size={meta['world_size']} "
            f"shard_optimizer={bool(meta['shard_optimizer'])}, this job is world_size={world_size} "
            f"shard_optimizer={bool(shard_optimizer)}. "
            "Resume needs the same number of GPUs and the same shard_optimizer setting."
        )


def finalize_resume_dir(directory: Path) -> Optional[Path]:
    """Point ``directory`` at the newest complete checkpoint.

    Rank 0 calls this before any rank reads the directory. A ``.partial`` tree
    only counts once ``meta.json`` is in it, which is written after every rank
    has finished its files.
    """
    directory = Path(directory)
    partial = _partial_dir(directory)
    prev = _prev_dir(directory)
    if _complete(partial):
        if directory.exists():
            if prev.exists():
                shutil.rmtree(prev)
            directory.rename(prev)
        partial.rename(directory)
        if prev.exists():
            shutil.rmtree(prev)
    elif not _complete(directory) and _complete(prev):
        prev.rename(directory)
    if _complete(directory):
        return directory
    return None


def _complete(directory: Path) -> bool:
    return (directory / "meta.json").is_file() and (directory / "model.safetensors").is_file()


def _partial_dir(directory: Path) -> Path:
    return directory.with_name(directory.name + ".partial")


def _prev_dir(directory: Path) -> Path:
    return directory.with_name(directory.name + ".prev")


def _local_optimizer(optimizer: torch.optim.Optimizer) -> Optimizer:
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        return optimizer.optim
    return optimizer


def _local_param_names(module: torch.nn.Module, optimizer: Optimizer) -> List[str]:
    by_id = {id(p): name for name, p in module.named_parameters()}
    names = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            try:
                names.append(by_id[id(param)])
            except KeyError as exc:
                raise RuntimeError("optimizer parameter is not in the module") from exc
    return names


def _optimizer_tensors(optimizer: Optimizer) -> Iterator[Tuple[str, torch.Tensor]]:
    params = [p for g in optimizer.param_groups for p in g["params"]]
    missing = sum(len(optimizer.state.get(p, {})) == 0 for p in params)
    if missing:
        raise RuntimeError(
            f"AdamW state is missing for {missing} of {len(params)} parameters. "
            "The resume checkpoint is written after a step, once moments exist."
        )
    index = {id(p): i for i, p in enumerate(params)}
    for param, state in optimizer.state.items():
        if not isinstance(param, torch.Tensor) or id(param) not in index:
            continue
        prefix = str(index[id(param)])
        for key, value in state.items():
            if torch.is_tensor(value):
                yield f"{prefix}.{key}", value
            else:
                raise RuntimeError(f"AdamW state[{key}] is not a tensor; cannot save {type(value)}")


def _load_module_(module: torch.nn.Module, path: Path) -> None:
    from safetensors import safe_open

    current = module.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        saved = set(handle.keys())
        if saved != set(current):
            missing = sorted(set(current) - saved)[:5]
            unexpected = sorted(saved - set(current))[:5]
            raise RuntimeError(f"resume weight mismatch: missing={missing} unexpected={unexpected}")
        for name, tensor in current.items():
            src = handle.get_tensor(name)
            if tuple(src.shape) != tuple(tensor.shape):
                raise RuntimeError(f"resume weight shape mismatch for {name}: {tuple(src.shape)} vs {tuple(tensor.shape)}")
            tensor.copy_(src)
            del src


def _load_optimizer_(optimizer: Optimizer, path: Path, names: List[str]) -> None:
    from safetensors import safe_open

    params = [p for g in optimizer.param_groups for p in g["params"]]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        saved_names = json.loads((handle.metadata() or {})["param_names"])
        if saved_names != names:
            raise RuntimeError(
                "resume AdamW shard does not match this rank's parameters. "
                "The model, world size, or shard_optimizer setting changed."
            )
        fields: Dict[int, List[Tuple[str, str]]] = {}
        for key in handle.keys():
            idx_str, field = key.split(".", 1)
            fields.setdefault(int(idx_str), []).append((field, key))
        if set(fields) != set(range(len(params))):
            raise RuntimeError(
                f"resume AdamW has state for indices {sorted(fields)}, this rank owns {len(params)} parameters"
            )
        for idx, param in enumerate(params):
            group = next(g for g in optimizer.param_groups if any(p is param for p in g["params"]))
            fake_groups = [{
                "params": [0],
                "fused": group.get("fused", False),
                "capturable": group.get("capturable", False),
            }]
            state = {}
            for field, key in fields[idx]:
                value = handle.get_tensor(key)
                state[field] = Optimizer._process_value_according_to_param_policy(
                    param, value, 0, fake_groups, field
                )
                del value
            optimizer.state[param] = state
            del state


def _stream_save(
    path: Path,
    tensors: Iterable[Tuple[str, torch.Tensor]],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Write a safetensors file from a stream so only one tensor is on the host at a time."""
    path = Path(path)
    payload = path.with_suffix(path.suffix + ".payload")
    header: Dict[str, object] = {}
    offset = 0
    with open(payload, "wb") as out:
        for name, tensor in tensors:
            dtype, shape, nbytes = _write_tensor(out, tensor)
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        out.flush()
        os.fsync(out.fileno())
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as out, open(payload, "rb") as src:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    payload.unlink()


def _write_tensor(out, tensor: torch.Tensor) -> Tuple[str, List[int], int]:
    try:
        dtype = _DTYPE_TO_STR[tensor.dtype]
    except KeyError as exc:
        raise RuntimeError(f"cannot save dtype {tensor.dtype}") from exc
    cloned = tensor.detach().to("cpu", non_blocking=False).contiguous()
    nbytes = cloned.numel() * cloned.element_size()
    if nbytes:
        # View of this one tensor, not a second copy of the whole checkpoint.
        buf = (ctypes.c_char * nbytes).from_address(cloned.data_ptr())
        out.write(buf)
        del buf
    shape = list(cloned.shape)
    del cloned
    return dtype, shape, nbytes


def _barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
