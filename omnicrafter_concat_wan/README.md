# OmniCrafter backbone from Wan weights + channel concat (perspective + panorama)

This is the [Wan-init SCoPE run](../omnicrafter_scope_wan/README.md) with Plücker channel concat, FrameCrafter's camera input, in place of SCoPE. It is that run's comparison, on CIS. The other run directories are imported unchanged.

| | value | from |
|---|---|---|
| backbone | Wan2.1-I2V-14B | OmniCrafter |
| init | Wan's weights. The r32 LoRA (`q,k,v,o,ffn.0,ffn.2`) starts at zero. The patch embedding is Wan's 36 channels plus 384 zero ray channels, all trained. Step 0 is exactly Wan | this run |
| camera | `plucker_raymap` with `patch=8`, concatenated to the latents: 6 × 64 values per latent pixel, pinhole or equirect. Last camera at the origin, unit mean camera distance | FrameCrafter |
| objective, encoders, data, optimizer, scale | as the Wan-init SCoPE run | scope-wan |

Trainable tensors: 800 LoRA + 2 patch embedding = 802, the layout of a FrameCrafter adapter.

Near depth cancels in this gauge. The config still passes Zhaoliang's tables because they decide which panorama scenes are kept.

## Files

- `concat_wan_config.py`: the run config, scope-wan's without the SCoPE fields.
- `concat_wan_model.py`: the training module, the 36→420 patch embedding expansion and the RayCrafter-batch adapter.
- `train.py`: the trainer, `omnicrafter_scope_wan/train.py` with the initialization and rays swapped.
- `configs/`: run configs, with CIS paths. The near-depth paths point into this checkout.
- `train_4gpu.sh`: the launcher, run on the GPU node. It uses scope-ff's accelerate config.
- `tests/`: CPU tests.

## Run

```bash
PY=/cis/home/dli90/miniconda3/envs/omnicrafter/bin/python
OMP_NUM_THREADS=4 $PY -m unittest discover -s omnicrafter_concat_wan/tests -v
$PY omnicrafter_concat_wan/train.py --config omnicrafter_concat_wan/configs/omnicrafter_mix_concat_wan_4gpu_50k.yaml --dry-run
bash omnicrafter_concat_wan/train_4gpu.sh
```

Rerunning the same command continues from `<output_dir>/resume`, as in the scope-ff run.

## Not covered

- **Evaluation and sampling.**
- **Loss comparisons with FLUX 3.** Compare runs by evaluation metrics, as in the SCoPE run's README.
