# OmniCrafter backbone from Wan weights + SCoPE (perspective + panorama)

This is the [scope-ff run](../omnicrafter_scopeff/README.md) with two changes:
- it starts from Wan's own weights instead of the published FrameCrafter adapter;
- the camera module is plain SCoPE, as in `configs/flux3_mix_scope.yaml` on `main`, instead of scope-ff.

SCoPE is the only camera input: no Plücker ray concatenation and no scope-ff. `omnicrafter_scopeff/`, `omnicrafter_neardepth/` and `omnicrafter_handoff/` are imported unchanged.

| | value | from |
|---|---|---|
| backbone | Wan2.1-I2V-14B | OmniCrafter |
| init | Wan's weights. The r32 LoRA (`q,k,v,o,ffn.0,ffn.2`) starts at zero, and the patch embedding is Wan's 36-channel one, trained. Step 0 is exactly Wan | this run |
| objective | flow matching from noise on the 3 query latents, uniform timestep | scope-ff run |
| encoders | per-frame VAE, neutral temporal RoPE, cached empty T5 context, T5/CLIP freed | scope-ff run |
| camera | SCoPE in all 40 self-attentions after Q/K norm + RoPE: the 7 released inputs through a 64-wide MLP, `plucker_eps` 1e-3, the log-scale gate shift and no V residual | flux3-scope |
| rays | `plucker_tokens` on near-depth-scaled c2w; refs sorted near→far, relative to the nearest ref | flux3-scope |
| data | dl3dv_points 288×512 and HM3D + JHU panoramas 256×512, half each (one kind per rank and step). Each query is black with p 0.5, otherwise it keeps 0–75% dropout | flux3-scope |
| optimizer | AdamW lr 1e-5, wd 0.01, clip 1.0, 500-step linear warmup then constant | flux3-scope |
| scale | DeepSpeed ZeRO-2 bf16, 4 GPUs × 1 set, gradient checkpointing, 50k steps | |

Trainable tensors: 800 LoRA + 2 patch embedding + 40 × 12 SCoPE = 1282.

## Differences from the FrameCrafter-initialized runs to expect

- **Validation loss starts at Wan's.** Early losses don't compare with the runs that start from FrameCrafter.
- **The LoRA learns from nothing at lr 1e-5.** That rate was chosen for fine-tuning a trained adapter and for flux3 parity. DiffSynth's own default is 1e-4.
- **`modify_channels` is off.** With it, `WanTrainingModule` replaces the patch embedding with a random one, and only a checkpoint would restore it. `train.py` unfreezes Wan's patch embedding instead, and asserts it matches the weights on disk.
- **Same random start on every rank.** The LoRA A matrices and SCoPE's MLPs are drawn from `seed`. `initialization/wan_step0.safetensors` holds that draw, and every rank asserts it starts from those weights.

## Files

- `scope_wan_config.py`: the run config. It has no FrameCrafter or scope-ff fields, so a YAML that sets them is rejected.
- `train.py`: the trainer, `omnicrafter_scopeff/train.py` with the initialization swapped. The model is scope-ff's `ScopeFFTrainingModule` with `dir_freqs=0, patch_size=False`.
- `configs/`: run configs.
- `train_4gpu.sbatch`: the Slurm launcher, on `rchella4_bengal`. It uses scope-ff's accelerate config.
- `tests/`: CPU tests.

## Run

```bash
PY=/weka/projects/rchella4/dli90/envs/omnicrafter/bin/python
OMP_NUM_THREADS=4 $PY -m unittest discover -s omnicrafter_scope_wan/tests -v
$PY omnicrafter_scope_wan/train.py --config omnicrafter_scope_wan/configs/omnicrafter_mix_scope_wan_4gpu_50k.yaml --dry-run
sbatch omnicrafter_scope_wan/train_4gpu.sbatch
```

The output directory has the same layout as the scope-ff run's, with `initialization/wan_step0.safetensors` in place of the converted FrameCrafter file. See [its README](../omnicrafter_scopeff/README.md).

## Variant: xliu253's task mix and LR start

`configs/omnicrafter_mix_scope_wan_xliutasks_4gpu_50k.yaml` is the same run with three settings from xliu253's mixed recipe (`omnicrafter_handoff/docs/EXPERIMENTS.md`):

| | base config | xliutasks |
|---|---|---|
| NVS | each query black with p 0.5 (`nvs_ratio`) | each set NVS with p 0.5, all three queries black together (`nvs_probability`) |
| cleanup dropout | 0–75% | 0–100%, so a cleanup query can be black too |
| LR start | 500-step linear warmup | `lr_schedule: constant_lr`: 1/3 of the LR for optimizer steps 1–2, then 1e-5 |

xliu's trainer prepares torch's `ConstantLR` (1/3 for 5 scheduler steps) with accelerate's defaults, which step it once per process per training step. On 4 GPUs the reduced LR therefore lasts 2 optimizer steps, not 5. `train.lr_lambda` reproduces that.

It runs on `rchella4_videolincs`:

```bash
CONFIG=omnicrafter_scope_wan/configs/omnicrafter_mix_scope_wan_xliutasks_4gpu_50k.yaml \
  sbatch -A rchella4_videolincs -J oc-scope-wan-xt-4g omnicrafter_scope_wan/train_4gpu.sbatch
```

Resubmit the same command to continue.

## Resume

As the scope-ff run: resubmit the same job after a walltime stop or a crash. It needs 4 GPUs, the same LoRA and SCoPE settings, the same `seed`, and a DeepSpeed version that reads the saved state. To train past `max_steps`, raise it and resubmit.

## Not covered

- **Evaluation and sampling.** Sampling with these adapters needs `install_scope_ff` (with `dir_freqs=0, patch_size=False`) and `model_fn_scope_ff` installed in the Wan pipeline first, and Wan's 36-channel patch embedding replaced by the saved one.
- **Loss comparisons with FLUX 3.** Its objective is bridge matching with LPIPS/L1, so its losses don't compare with these. Compare runs by evaluation metrics.
