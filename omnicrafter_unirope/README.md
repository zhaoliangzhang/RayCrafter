# OmniCrafter + UniRoPE v3 (perspective + panorama)

This is the [scope-ff run](../omnicrafter_scopeff/README.md) with one change: the camera module is RayCrafter's UniRoPE v3 from [`configs/flux3_mix_unirope_v3.yaml`](../configs/flux3_mix_unirope_v3.yaml), in place of scope-ff.

UniRoPE is the only camera input: no Plücker ray concatenation and no SCoPE. `omnicrafter_scopeff/`, `omnicrafter_neardepth/` and `omnicrafter_handoff/` are imported unchanged.

| | value | from |
|---|---|---|
| backbone / init | Wan2.1-I2V-14B with the published FrameCrafter LoRA (r32, `q,k,v,o,ffn.0,ffn.2`); patch embedding cut to its 36 image channels, trained | scope-ff run |
| objective | flow matching from noise on the 3 query latents, uniform timestep | scope-ff run |
| encoders | per-frame VAE, cached empty T5 context, T5/CLIP freed | scope-ff run |
| camera | `unirope_attention` in all 40 self-attentions. It takes the whole 128-d head: camera center plus direction and log distance of three patch-corner points at depth, 15 coordinates × 4 frequencies = 120 channels, 8 pass through. Wan's RoPE isn't applied. V and the attention output are rotated too, and attention runs once per camera (5 passes per layer) | flux3-unirope-v3 |
| direction band | per query camera, from its angular patch size (`unirope_adaptive_freqs`) | flux3-unirope-v3 |
| depth | a zero-initialized `DepthHead` per block on the modulated attention input, with its own AdamW group: lr 4e-4, wd 0.5 | flux3-unirope-v3 |
| poses | refs sorted near→far, relative to the nearest ref, second ref at unit distance. The near-depth files only decide which scenes are dropped, so the training set matches the other runs | flux3-unirope-v3 |
| data | dl3dv_points 288×512 and HM3D + JHU panoramas 256×512, half each (one kind per rank and step). Each query is black with p 0.5, otherwise it keeps 0–75% dropout | scope-ff run |
| optimizer | AdamW lr 1e-5, wd 0.01, clip 1.0, 500-step linear warmup then constant | scope-ff run |
| scale | DeepSpeed ZeRO-2 bf16, 4 GPUs × 1 set, gradient checkpointing, 50k steps | |

Trainable tensors: 800 LoRA + 2 patch embedding + 40 × 2 depth head = 882.

## Differences from the scope-ff run to expect

- **Not identity at step 0.** scope-ff starts as exactly the FrameCrafter model. UniRoPE replaces Wan's RoPE and rotates V and the output, so step-0 validation loss is higher and early losses don't compare.
- **No `torch.compile`.** `train_4gpu.sbatch` sets `TORCH_COMPILE_DISABLE=1`, so the `@torch.compile` functions in `raycrafter.cameras` run eagerly. Triton builds a C stub against `Python.h`, which this env's system Python doesn't have.

## Files

- `unirope_wan.py`: `install_unirope`, `model_fn_unirope`.
- `unirope_model.py`: `UniRoPETrainingModule`, `optimizer_groups`, and `to_omnicrafter_unirope`, which converts a RayCrafter batch into an OmniCrafter set.
- `train.py`: the trainer, `omnicrafter_scopeff/train.py` with the camera module swapped.
- `configs/`: run configs.
- `train_4gpu.sbatch`: the Slurm launcher, on `rchella4_bengal`. It uses scope-ff's accelerate config.
- `tests/`: wiring tests, which run on the GPU if there is one.

## Run

```bash
PY=/weka/projects/rchella4/dli90/envs/omnicrafter/bin/python
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=4 $PY -m unittest discover -s omnicrafter_unirope/tests -v
TORCH_COMPILE_DISABLE=1 $PY omnicrafter_unirope/train.py --config omnicrafter_unirope/configs/omnicrafter_mix_unirope_v3_4gpu_50k.yaml --dry-run
sbatch omnicrafter_unirope/train_4gpu.sbatch
```

The output directory `output_dir` holds:
- `step-N.safetensors`: bf16 trainable weights, written every `save_interval` and at the end.
- `resume/`: the full state. It's rewritten every `resume_interval`, at `time_limit_hours` and at the end, by an atomic directory swap. It contains:
  - the DeepSpeed model states (trainable only);
  - the per-rank ZeRO-2 optimizer shards, with fp32 masters and AdamW moments for both groups;
  - the LR scheduler and per-rank RNG;
  - `meta.json`, holding the step, world size, settings fingerprint and a checksum.
- `resume_step<max_steps>/`: hard links to the final full state.
- `reports/`: `train_loss.jsonl` (every step), `train_log.jsonl` (train and val metrics) and `trainable_parameters.json`.
- `wandb/`: offline runs. Sync them later with `wandb sync <output_dir>/wandb/offline-run-*`.

To confirm the depth heads are learning, check the startup line `depth heads: 80 tensors, lr 0.0004, weight decay 0.5`, then `depth/log_depth_bias` and `depth/sigma_bias` in the logs. `lr_depth` is 40 × `lr` throughout.

## Resume

- **After a walltime stop or a crash:** resubmit the same job. It loads `resume/`, checks the checksum and continues. The data order is a fresh shuffle from the restored RNG.
- **To train past `max_steps`:** raise `max_steps` and resubmit. The LR is constant after warmup, so there is no schedule jump.
- **To branch from the final state:** run `cp -al <out>/resume_step50000 <new_out>/resume`, then point a config at `<new_out>`.
- **Resume needs:**
  - 4 GPUs;
  - the same LoRA and UniRoPE settings and depth-head optimizer settings, which are checked against `meta.json`;
  - the same FrameCrafter initialization;
  - a DeepSpeed version that reads the saved state.

## Not covered

- **Evaluation and sampling.** Sampling with these adapters needs `install_unirope` and `model_fn_unirope` installed in the Wan pipeline first.
- **Loss comparisons with FLUX 3.** Its objective is bridge matching with LPIPS/L1, so its losses don't compare with these. Compare runs by evaluation metrics.
