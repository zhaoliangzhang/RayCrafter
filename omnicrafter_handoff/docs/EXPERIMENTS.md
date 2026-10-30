# Model identity and experiment contract

## Names that are not interchangeable

- **Original FrameCrafter:** upstream model and our older two-reference,
  one-target NVS adaptation. The HM3D-only original 100k checkpoint is not the
  current mixed model.
- **Model 2 = OmniCrafter:** reference images plus posed target-image slots,
  supporting black-input NVS and noisy-render cleanup. Standard panorama recipe
  is 2ref3; separate 2ref1 and 2ref10 experiments also exist.
- **OmniRayCrafter / SCoPE experiment:** different camera injection through
  attention. It has a different checkpoint structure. A parent directory named
  `omniraycrafter` does **not** imply every child run uses SCoPE.
- **Current mixed-projection run:** original OmniCrafter **concatenated-ray**
  conditioning, with ERP/perspective-aware ray generation. Starts from published
  FrameCrafter adapter initialization, not from our 650k teacher or a SCoPE checkpoint.
- **Mixed-projection SCoPE variant:** the same data and training recipe with
  Q/K-only SCoPE camera injection, explicitly selected as `architecture: "scope"`.
  Its input patch has 36 image/conditioning channels instead of 420 image+ray
  channels. The October 29 portable training update supports this variant.
- **Distilled student:** a separately trained two-step model. Setting a teacher's
  sampler to two steps does not reproduce this student. Not trained by these wrappers.
- **SyncFix and FixAnything:** separate model/training pipelines, not interchangeable checkpoints.

## Current mixed-projection recipe

| Field | Actual setting |
| --- | --- |
| Data | 200k logical entries: DL3DV100k / HM3D60k / JHU40k |
| Projection ratio | Perspective:panorama = 1:1 |
| Panorama-only ratio | HM3D:JHU = 6:4, **not** 1:1 |
| Scene balance | All 3,552 HM3D and 148 non-held-out JHU sequence folders covered |
| Slots | `[ref0, ref1, query0, query1, query2]`; two clean refs, three outputs |
| Resolution W×H | ERP 512×256; perspective 512×288 |
| Explicit NVS switch | 0.5 **per sample**; all three query conditions become black together |
| Cleanup branch | Stored noisy RGBs, independent online per-query pixel dropout at integer rates 0–100% inclusive |
| Pose conditioning | Poses for all five views; never remove target poses for NVS |
| Initialization | Published FrameCrafter adapter + Wan2.1-I2V-14B backbone |
| Trainable tensors | Rank32 DiT LoRA (`q,k,v,o,ffn.0,ffn.2`) + patch embedding, 802 tensors |
| Frozen | Remaining backbone and VAE/text/image encoders |
| Loss | Flow-matching, only the final three target frames supervised |
| Optimization | AdamW on GPU, nominal lr `1e-5`, weight decay `0.01` |
| Distribution | 4 GPUs, true batch1/GPU, accumulation1, global batch4 |
| Precision/memory | bf16, CPU model initialization, ZeRO2, no optimizer/parameter CPU offload, gradient checkpointing |
| Endpoint/checkpoints | Cumulative 300k updates, save every2k in original run |

The optimizer uses `ConstantLR(optimizer)` with PyTorch's defaults: its initial
factor/interval are not a custom constant-1.0 scheduler. The initial reduced-LR
phase is restarted when launching a weights-only continuation. Preserve this
code when reproducing the run; changing it constitutes a recipe change.

Since the cleanup augmentation includes 100%, the explicit NVS switch being
0.5 does **not** imply exactly 50% of individual query images are non-black.
Cleanup-tagged queries can also become black, and stored priors already contain
holes. Task labels are diagnostics, not extra DiT inputs.

## Per-query versus per-sample task mixture

The older panorama dataloader has two different parameters:

| Control | Behavior |
| --- | --- |
| `nvs_ratio=0.5`, `nvs_probability=0` | Independent 50% black event for **each query**; mixed black/noisy groups allowed |
| `nvs_ratio=0`, `nvs_probability=0.5` | 50% **sample-level** event; all three black together |

The 650k teacher continuation uses the first control, with cleanup dropout capped
at 99% for that explicit per-query recipe. The current mixed implementation uses
the second, sample-level control and cleanup dropout through 100%. Do not transfer
the teacher's shorthand “50/50” to the current run without documenting this difference.
The older portable panorama package supports both; the new mixed wrapper fixes
the current behavior and does not silently accept unsupported mixture knobs.

## Camera and image conventions

Every view carries **OpenCV world-to-camera** 4×4 poses in a shared world frame:
`p_camera = R @ p_world + t`. Camera axes are right/down/forward. Camera center is
`-R.T @ t`, not `t`. Input pose units must be consistent within a group.

All five cameras are normalized together: the **last query camera** becomes the
origin, and translations are scaled by the mean camera-center distance. Query
ordering therefore matters. Single-query and three-query grouping can change
normalization/attention context even if an individual target image is identical.

ERP rays use the full spherical projection; do not crop or interpret panoramas
as pinhole images. Perspective rays use per-view scaled intrinsics and invert
the supplied OpenCV radial/tangential distortion. Do not undistort RGB in addition
to this ray calculation unless the entire calibration/prior protocol is changed.

Plücker feature order is **`[d, o × d]`**, not the reversed convention. Six channels
at full image resolution become 384 channels with PixelUnshuffle8, concatenated
with 36 WAN latent/conditioning channels before patch embedding: **420 channels**.
Typical packed DiT input is `[B,420,5,H/8,W/8]`, namely `[B,420,5,32,64]` for ERP
and `[B,420,5,36,64]` for perspective. Rays are not extra image tokens. The local
implementation independently VAE-encodes frames and neutralizes temporal RoPE
in this conditioning path; spatial positional encoding remains.

The text encoder is still loaded to supply the empty-prompt context. It is not
evidence of a separate text-training task. Ground-truth target RGB/depth is not
an inference condition. No separate depth image is an input to this RGB model.

### SCoPE-specific camera path

The bundled `snapshots/mixed/scope_model.py` uses the vendored SCoPE
Normalize-Gate-Inject implementation (commit
`2fa984036004fb564fc2b0e888ce2bd95ab03735`). Pluecker rays are evaluated at patch
centers with stride 16: 2,560 six-component rays per five-view panorama group,
or 2,880 per perspective group. Perspective intrinsics/distortion and the common
last-query pose gauge remain the same as for the concat experiment.

SCoPE injects learned geometry into self-attention Q/K after the existing Q/K
normalization and spatial RoPE. There is no V camera residual, no concatenated
ray channels, and no RayRoPE. Temporal RoPE remains neutral. The DiT input is
`[B,36,5,H/8,W/8]`, plus explicit `[B,tokens,6]` rays. New SCoPE geometry projection
weights are zero-initialized; the learned gate and normalization parameters use
the bundled implementation's initialization. All 400 SCoPE parameter tensors
are trained alongside the 802 existing LoRA/patch tensors.

For published initialization, the portable launcher writes a **new** adapter
under the output directory: retain all LoRA/bias values and slice the published
patch weight to `[:, :36]`. The published file is never modified. This image-only
802-tensor initialization is not a trained 1,202-tensor SCoPE checkpoint.

## Checkpoints, steps and continuation

Our `.safetensors` files contain LoRA/patch weights, **not the 14B backbone** and
not a DeepSpeed training-state checkpoint. You need the matching WAN and tokenizer
files as well. The concat adapter contains 802 trainable tensors; the Q/K-only
SCoPE adapter contains 1,202. The training launcher validates against the chosen
architecture, including all SCoPE parameter names/shapes. Cross-architecture
resume is rejected. The portable evaluation launcher remains concat-only.

Loading `step-28000` and ending at `300000` performs **272,000 additional updates**.
The portable mixed config uses `initial_step` and **cumulative** `final_step`.
The older `panorama/scripts/train.py --steps` wrapper instead means **additional**
steps. Read the relevant CLI; these flags are not interchangeable.

Weights-only continuation resets Adam moments, scheduler, RNG and data-loader
position. The seed may be the same, but the optimizer trajectory is not a bitwise
resume. If the new cluster requires exact restart, implement and verify complete
optimizer/scheduler/RNG/sampler state saving as a separately versioned change.

The portable mixed trainer intentionally writes to a **new output directory**.
Use a specific checkpoint, never an ambiguous “latest” glob, and preserve its
SHA256 and ancestry. No existing checkpoint is overwritten.

## Memory, speed and scheduling knowledge

CPU initialization is different from CPU optimizer offload: this recipe loads
models initially on CPU but verifies Adam moment tensors are on CUDA. Seven or
eight inference workers can also consume substantial host RAM; model loading
is serialized with a file lock to reduce simultaneous host-memory peaks.

The old io104 job optionally parked unused memory in inert GPU buffers at the
user's request. **That behavior is disabled in the portable launcher.** It is not
part of the model and is not an exclusive resource lock. Use scheduler GPU
allocations on the new cluster, rather than trying to reserve cards with tensors.

Stage model weights and data onto fast storage. The old mixed training had NFS
image-read stalls; DDP peers then timed out during collectives. A timeout is not
automatically GPU OOM, and an idle-looking GPU does not establish that loading
has hung. Retain per-rank logs, system RAM/GPU snapshots and data-read diagnostics.

The restarted io95 evaluation previously failed when other processes occupied
about13GiB on cards already running ~31GiB inference workers. Idle polling is not
an exclusive allocation. New-cluster launchers respect `CUDA_VISIBLE_DEVICES`
and never kill, claim, or reserve another user's processes.
