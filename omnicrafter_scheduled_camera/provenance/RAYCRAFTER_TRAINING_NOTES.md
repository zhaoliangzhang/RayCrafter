# 相机编码训练注意事项

适用于 FLUX 3 Klein Base 上的 SCoPE（含 scope-ff）、RayRoPE、UniRoPE，数据为 dl3dv_points 单独训练或 dl3dv_points + 全景（framecrafter_5input）混训（`flux3_mix_*`）。代码以 main 上的 1cd10d2 为准。

## SCoPE：near-depth 文件

SCoPE 要求相机平移以场景的 near depth 为单位：每个场景的 c2w 平移除以该场景的 `near_depth`，让 COLMAP 尺度各不相同的场景有统一的运动幅度（SCoPE 原论文的归一化）。不设 near-depth 时平移保持原始尺度，SCoPE 的 Plücker moment 量级会乱。

config 里两个字段：

| 字段 | 作用于 | 生成脚本 |
| --- | --- | --- |
| `near_depth_json` | dl3dv_points | `scripts/estimate_near_depth_dl3dv_points.py`（`scripts/near_depth_dl3dv_a100.sbatch`） |
| `panorama_near_depth_json` | 全景混训数据 | `scripts/estimate_near_depth_panorama.py`（`scripts/near_depth_panorama_a100.sbatch`） |

- 文件格式：`{"<scene_name>": {"near_depth": 1.23, "n_pairs_total": 20, ...}}`，由 `data/dl3dv_points.py:load_near_depth` 读取。
- 估计方法：RAFT 光流 + 前后向一致性检查，用已知相对位姿三角化，`near_depth` = 各帧对 25% 分位深度的中位数。全景版本用 equirect 像素射线代替 `K⁻¹`，只用 ±45° 纬度带。
- **没有估计值的场景会被直接丢弃**，日志里会打印 `near depth from ... (N dropped)`，训练开始时检查一下 N。
- 全景不设 `panorama_near_depth_json` 时，平移会改为除以 5 个相机到最后一个 query 相机的平均距离（OmniCrafter 的规范）。这和 dl3dv 的 near-depth 单位不一致，**混训 SCoPE 时两个字段都要设**。
- RayRoPE / UniRoPE / plucker_concat 会自己重新归一化位姿，不依赖 near depth；但它们的 config 也设了 `near_depth_json`，这样被丢弃的场景和 SCoPE 一致，训练集相同、结果可比。新建 config 时保留这一行。
- 评测（`eval_ckpt.py`）用仓库内的相对路径 `near_depth/dl3dv_points.json` 和 `near_depth/framecrafter_5input.json`。
- scope-ff 在此之上加了三个开关：`scope_dir_freqs: 8`（方向的 Fourier 特征）、`scope_patch_size: true`（patch 角大小作为第 7 个输入通道）、`scope_mlp_hidden: 64`。不设时默认 0 / false / 64，即原版 SCoPE，老 ckpt 照常加载。

## RayRoPE：全景数据的深度符号

参考实现的 RayRoPE 在 query 相机里用 `|z| / w` 作为深度坐标（代码里是 `sqrt(z² + 1e-9)`，在 z = 0 处梯度平滑）。对针孔相机没问题：可见的点都在相机前方。

全景的一半像素在任何相机的后方。若仍取 `|z|`，相机前后对称的两个点会得到相同的深度编码，前后无法区分。所以加入全景数据后去掉了绝对值：

- `src/raycrafter/cameras/rayrope.py` 的 `_query_positions(..., signed_depth)`：`signed_depth=True` 时用带符号的 `z / w`。
- `signed_depth = camera_model == "equirect"`：**只有全景 batch 去掉绝对值，针孔 batch 仍是参考实现的 `|z|`**。混训时每个 batch 只有一种相机模型（由 `panorama_ratio` 决定），两种 batch 各走各的。
- 深度会被 clamp 到 `[-MAX_D_F, MAX_D_F]`，频率范围按此对称区间设置。
- 因为针孔路径没变，用 dl3dv_points 单独训练的老 RayRoPE ckpt 行为不变；但它们没见过带符号的深度，不要直接拿来评测全景。

UniRoPE 编码的是点的方向和 log 距离（距离总为正，方向本身带前后信息），没有这个问题。

## RayRoPE 和 UniRoPE：depth predict head 的单独学习率

两者每个 block 都有一个零初始化的 `DepthHead`，输出 log depth 和 sigma（初始 log depth = 0、sigma = 3，即"深度未知"）。如果和 backbone 共用 1e-5 的学习率，深度头几乎不动：dl3dv RayRoPE 跑了 90k step 后 sigma 仍约 2.96。

所以深度头放在单独的 AdamW 参数组里（`train.py`），config 里用：

```yaml
depth_head_lr: 4.0e-4          # 参考 RayRoPE 的设置
depth_head_weight_decay: 0.5
```

- 不设（`null`）时退回 `learning_rate` / `weight_decay`，也就是上面那个学不动的情况。
- 深度头学习率按 `lr_scale = depth_head_lr / learning_rate` 跟随 backbone 的调度（warmup、cosine 都同比例缩放）。
- 训练开始时日志会打印 `depth heads: N tensors, lr 0.0004, weight decay 0.5`，以此确认设置生效。
- 训练中看 `depth/log_depth_bias`、`depth/sigma_bias` 两个指标，确认深度头确实在学。
- 已经设了的 config：`flux3_dl3dv_rayrope`、`flux3_dl3dv_unirope_fp`、`flux3_mix_rayrope`、`flux3_mix_unirope`、`flux3_mix_unirope_v3`。
- **还没设的旧 config**：`flux3_dl3dv_unirope`、`flux3_dl3dv_unirope_anchor`、`klein9b_dl3dv_rayrope`、`klein9b_dl3dv_scope_rayrope`。复用它们之前先补上这两行。

## 文件位置

| 内容 | 当前服务器（JHU） | CIS（r32） |
| --- | --- | --- |
| dl3dv near depth | `/home/jhu/zzhan288/projects_rchella4/zzhan288/data/dl3dv_points/near_depth.json`（1119 个场景） | `/cis/home/zzhan288/RayCrafter/near_depth/dl3dv_points.json` |
| 全景 near depth | `/home/jhu/zzhan288/projects_rchella4/zzhan288/data/framecrafter_5input/near_depth.json`（3710 个场景） | `/cis/home/zzhan288/RayCrafter/near_depth/framecrafter_5input.json` |
| dl3dv_points 数据 | `/home/jhu/zzhan288/projects_rchella4/zzhan288/data/dl3dv_points/{train,test}` | 已在 CIS 上，见 `scripts/eval_mix_cis.sh` |
| ckpt | `outputs/<run>/transformer_step<N>.safetensors` | `/cis/home/zzhan288/RayCrafter/outputs/<run>/` |

两个 near-depth 文件都在当前服务器上估计，由 `scripts/rsync_to_cis.sbatch` 复制到 CIS（CIS 上不重新估计）。
