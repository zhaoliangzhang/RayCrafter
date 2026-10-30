# OmniCrafter + scope-ff (perspective + panorama)

This is xliu253's OmniCrafter, the [near-depth mixed run](../omnicrafter_neardepth/README.md), with two changes:
- the camera module is RayCrafter's scope-ff from [`configs/flux3_mix_scope_ff.yaml`](../configs/flux3_mix_scope_ff.yaml), in place of the plain SCoPE module;
- it trains on the flux3 mix data and schedule, read directly from Skipjack's copies, so the 747 GB CIS image cache isn't needed.

There is still no Plücker ray concatenation. `omnicrafter_neardepth/` and `omnicrafter_handoff/` are imported unchanged.

| | value | from |
|---|---|---|
| backbone / init | Wan2.1-I2V-14B with the published FrameCrafter LoRA (r32, `q,k,v,o,ffn.0,ffn.2`); patch embedding cut to its 36 image channels, trained | neardepth run |
| objective | flow matching from noise on the 3 query latents, uniform timestep | neardepth run |
| encoders | per-frame VAE, neutral temporal RoPE, cached empty T5 context, T5/CLIP freed | neardepth run |
| camera | scope-ff in all 40 self-attentions after Q/K norm + RoPE. Its inputs are the 7 released ones, 48 sin/cos direction features and the patch's angular size, through a 64-wide MLP. Also `plucker_eps` 1e-3, the log-scale gate shift and no V residual | flux3-scope-ff |
| rays | `plucker_tokens(..., patch_size=True)` on near-depth-scaled c2w; refs sorted near→far, relative to the nearest ref | flux3-scope-ff |
| data | dl3dv_points 288×512 and HM3D + JHU panoramas 256×512, half each (one kind per rank and step). Each query is black with p 0.5, otherwise it keeps 0–75% dropout | flux3-scope-ff |
| optimizer | AdamW lr 1e-5, wd 0.01, clip 1.0, 500-step linear warmup then constant | flux3-scope-ff |
| scale | DeepSpeed ZeRO-2 bf16, 4 GPUs × 1 set, gradient checkpointing, 50k steps | |

Trainable tensors: 800 LoRA + 2 patch embedding + 40 × 12 scope-ff = 1282.

## Files

- `scope_ff.py`: `install_scope_ff`, `model_fn_scope_ff`.
- `scopeff_model.py`: `ScopeFFTrainingModule`, and `to_omnicrafter`, which converts a RayCrafter batch into an OmniCrafter set.
- `train.py`: the trainer.
- `configs/`: run configs.
- `train_4gpu.sbatch`: the Slurm launcher.
- `accelerate_zero2.yaml`: the accelerate config. It points at the snapshot's `deepspeed_zero2.json`.
- `tests/`: CPU tests.

## Run

```bash
# Environment: torch 2.8.0+cu128, accelerate 1.14.0, deepspeed 0.19.5, transformers 5.15.1, peft 0.20.0,
# plus opencv/sentencepiece/ftfy/modelscope/imageio/pandas/datasets/webdataset/wandb.
PY=/weka/projects/rchella4/dli90/envs/omnicrafter/bin/python
$PY -m unittest discover -s omnicrafter_scopeff/tests -v
$PY omnicrafter_scopeff/train.py --config omnicrafter_scopeff/configs/omnicrafter_mix_scope_ff_4gpu_50k.yaml --dry-run
sbatch omnicrafter_scopeff/train_4gpu.sbatch
```

The output directory `output_dir` holds:
- `step-N.safetensors`: bf16 trainable weights, written every `save_interval` and at the end.
- `resume/`: the full state. It's rewritten every `resume_interval`, at `time_limit_hours` and at the end, by an atomic directory swap. It contains:
  - the DeepSpeed model states (trainable only);
  - the per-rank ZeRO-2 optimizer shards, with fp32 masters and AdamW moments;
  - the LR scheduler and per-rank RNG;
  - `meta.json`, holding the step, world size, settings fingerprint and a checksum.
- `resume_step<max_steps>/`: hard links to the final full state.
- `reports/`: `train_loss.jsonl` (every step), `train_log.jsonl` (train and val metrics) and `trainable_parameters.json`.
- `wandb/`: offline runs. Sync them later with `wandb sync <output_dir>/wandb/offline-run-*`.

## Resume

- **After a walltime stop or a crash:** resubmit the same job. It loads `resume/`, checks the checksum and continues. The data order is a fresh shuffle from the restored RNG.
- **To train past `max_steps`:** raise `max_steps` and resubmit. The LR is constant after warmup, so there is no schedule jump.
- **To branch from the final state:** run `cp -al <out>/resume_step50000 <new_out>/resume`, then point a config at `<new_out>`.
- **Resume needs:**
  - 4 GPUs;
  - the same LoRA and scope-ff settings, which are checked against `meta.json`;
  - the same FrameCrafter initialization;
  - a DeepSpeed version that reads the saved state.

## Not covered

- **Evaluation and sampling.** The handoff's evaluator is concat-only. Sampling with these adapters needs `install_scope_ff` and `model_fn_scope_ff` installed in the Wan pipeline first.
- **Loss comparisons with FLUX 3.** Its objective is bridge matching with LPIPS/L1, so its losses don't compare with these. Compare runs by evaluation metrics.
