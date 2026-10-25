"""Multi-view bridge-matching model: refine / synthesize a target view from refs + cameras.

Token sequence (one flat self-attention sequence, no channel surgery):

    [ ref_1 … ref_R | view_1 … view_V ]

* Views travel the bridge ``z_t = t·z_src + (1-t)·z_tgt`` (FLUX convention:
  t=1 is the source, t=0 is clean). ``z_src`` is a degraded render, or pure
  Gaussian noise for novel-view synthesis (``noise_space=latent``, alpha=1).
* Refs are clean and stay clean at every t.
* Cameras are ordered ``[ref..., views...]`` and enter only through the
  camera modules (SCoPE on Q/K/V, RayRoPE or PRoPE on the RoPE channels).

The loss is MSE on the velocity ``z_src - z_tgt`` of the V view tokens, plus
optional LPIPS / L1 on the decoded one-step estimate ``x0 = z_t - t·v``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_sft
from torch.utils.checkpoint import checkpoint

from .backbones import flux2, flux3
from .config import Config
from .lora import add_lora, freeze_base_model, load_trainable_weights

BACKBONES = {"flux2": flux2, "flux3": flux3}


class RayCrafterModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self._dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
        self.backbone = backbone = BACKBONES[cfg.backbone]

        self.ae = backbone.load_vae(cfg.ae_path, device)
        self.ae.requires_grad_(False)

        self.transformer = backbone.load_backbone(cfg.model_variant, cfg.model_path, device)

        if cfg.use_lora:
            n = len(add_lora(self.transformer, cfg.lora_target_modules,
                             cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout))
            freeze_base_model(self.transformer)
            print(f"LoRA on {n} layers (rank={cfg.lora_rank})")

        # Added after freeze_base_model, so camera modules always train.
        backbone.install_camera_attention(
            self.transformer, scope=cfg.uses_scope, rope_type=cfg.rope_type, apply_vo=cfg.rope_apply_vo
        )
        print(f"camera_condition={cfg.camera_condition}  rope_type={cfg.rope_type}")

        if cfg.resume_from:
            if cfg.use_lora:
                load_trainable_weights(self.transformer, cfg.resume_from)
            else:
                # Read on the host and copy into the loaded weights: the GPU never
                # holds two copies of the transformer (22 GB would not fit a 24 GB card).
                sd = load_sft(cfg.resume_from, device="cpu")
                self.transformer.load_state_dict(sd, strict=True)
                del sd
            print(f"Resumed from {cfg.resume_from}")

        if cfg.gradient_checkpointing:
            backbone.enable_gradient_checkpointing(self.transformer)

        self.lpips = None
        if cfg.lpips_loss_weight > 0:
            import lpips

            self.lpips = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "RayCrafterModel":
        """Only the transformer switches mode. LPIPS has Dropout(0.5) in its
        linear heads, which ``train()`` would otherwise turn on."""
        super().train(mode)
        self.ae.eval()
        if self.lpips is not None:
            self.lpips.eval()
        return self

    # ── VAE ───────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` in [-1, 1] → ``(N, 128, H/16, W/16)``; center-crops to a multiple of 16."""
        H, W = images.shape[-2:]
        P = self.backbone.PATCH
        H16, W16 = (H // P) * P, (W // P) * P
        if (H16, W16) != (H, W):
            top, left = (H - H16) // 2, (W - W16) // 2
            images = images[..., top:top + H16, left:left + W16]
        return self.ae.encode(images.float()).to(self._dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z.float()).clamp(-1, 1)

    def _encode_views(self, imgs: torch.Tensor) -> torch.Tensor:
        B, V = imgs.shape[:2]
        z = self.encode(imgs.flatten(0, 1))
        return z.reshape(B, V, *z.shape[1:])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sort_refs(self, ref_imgs, c2w, fxfycxcy):
        """Reorder refs near→far by camera-center distance to the views (mean over views)."""
        if not self.cfg.sort_refs_by_distance:
            return ref_imgs, c2w, fxfycxcy
        R = ref_imgs.shape[1]
        centers = c2w[:, :, :3, 3].float()
        order = torch.cdist(centers[:, :R], centers[:, R:]).mean(dim=-1).argsort(dim=1)
        b = torch.arange(ref_imgs.shape[0], device=order.device)[:, None]
        cam_order = torch.cat(
            [order, torch.arange(R, c2w.shape[1], device=order.device).expand(order.shape[0], -1)], dim=1
        )
        return ref_imgs[b, order.to(ref_imgs.device)], c2w[b, cam_order], fxfycxcy[b, cam_order]

    def _check_cameras(self, c2w, fxfycxcy, num_refs: int, num_views: int) -> None:
        if not self.cfg.use_pose:
            return
        if c2w is None or fxfycxcy is None:
            raise ValueError("this config needs camera_c2w / camera_fxfycxcy (pose.json in the shards)")
        if c2w.shape[1] != num_refs + num_views:
            raise ValueError(
                f"expected R+V={num_refs + num_views} cameras ordered [ref..., views...], got {c2w.shape[1]}"
            )

    def _source_latent(self, z_src, z_tgt, noise_alpha):
        """noise_space=latent: z_src = (1-a)·z_tgt + a·N(0, I) where a > 0 (a=1 needs no z_tgt)."""
        if self.cfg.noise_space != "latent" or noise_alpha is None:
            return z_src
        a = noise_alpha.to(device=z_src.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
        if not bool((a > 0).any()):
            return z_src
        if z_tgt is None and bool(((a > 0) & (a < 1)).any()):
            raise ValueError("partial latent noise (0 < alpha < 1) needs the clean target")
        eps = torch.randn(z_src.shape, device=z_src.device, dtype=torch.float32)
        base = z_tgt.float() if z_tgt is not None else torch.zeros_like(eps)
        mixed = (1.0 - a) * base + a * eps
        return torch.where(a > 0, mixed, z_src.float()).to(z_src.dtype)

    def _prepare(self, src_imgs, ref_imgs, c2w, fxfycxcy, noise_alpha, tgt_imgs, camera_model="pinhole"):
        """Shared by training and sampling: sort refs, encode, build ref tokens + camera context."""
        B, V = src_imgs.shape[:2]
        R = ref_imgs.shape[1]
        H, W = src_imgs.shape[-2:]
        self._check_cameras(c2w, fxfycxcy, R, V)
        ref_imgs, c2w, fxfycxcy = self._sort_refs(ref_imgs, c2w, fxfycxcy)

        z_src = self._encode_views(src_imgs)
        z_tgt = self._encode_views(tgt_imgs) if tgt_imgs is not None else None
        z_ref = self._encode_views(ref_imgs)
        z_src = self._source_latent(z_src, z_tgt, noise_alpha)
        h, w = z_src.shape[-2:]

        start, step = self.backbone.ref_index(V, self.cfg.ref_t_scale)
        x_ref, ref_ids = self.backbone.pack_views(z_ref, start, step)
        cams = None
        if self.cfg.needs_cameras:
            cams = self.backbone.camera_inputs(
                c2w, fxfycxcy, (H, W), (h, w), self.cfg.uses_scope, self.cfg.rope_type,
                footprint=self.cfg.unirope_footprint, camera_model=camera_model,
            )
        return z_src, z_tgt, x_ref.to(self._dtype), ref_ids.to(src_imgs.device), cams, (h, w)

    def _forward(self, x_ref, ref_ids, x_views, view_ids, t, cams) -> torch.Tensor:
        """One transformer pass; returns the predicted velocity of the view tokens."""
        self.backbone.set_camera_context(self.transformer, cams)
        # Autocast lets training keep fp32 master weights while matmuls run in bf16.
        # No cast cache: each weight is used once per forward, and the cache would
        # hold a bf16 copy of every weight until the forward ends (~10 GB for FLUX 3).
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._dtype == torch.bfloat16,
                            cache_enabled=False):
            return self.backbone.run_transformer(
                self.transformer, x_ref, ref_ids, x_views, view_ids, t.to(self._dtype)
            )

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(
        self,
        src_imgs: torch.Tensor,                          # (B, V, 3, H, W) in [-1, 1]
        tgt_imgs: torch.Tensor,                          # (B, V, 3, H, W)
        ref_imgs: torch.Tensor,                          # (B, R, 3, H, W)
        camera_c2w: Optional[torch.Tensor] = None,       # (B, R+V, 4, 4)
        camera_fxfycxcy: Optional[torch.Tensor] = None,  # (B, R+V, 4) pixels
        noise_alpha: Optional[torch.Tensor] = None,      # (B,)
        camera_model: str = "pinhole",                   # one per batch: pinhole | equirect
    ) -> Dict[str, torch.Tensor]:
        B, V = src_imgs.shape[:2]
        device = src_imgs.device
        with torch.no_grad():
            z_src, z_tgt, x_ref, ref_ids, cams, (h, w) = self._prepare(
                src_imgs, ref_imgs, camera_c2w, camera_fxfycxcy, noise_alpha, tgt_imgs, camera_model
            )

        cfg = self.cfg
        t = torch.rand(B, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        if cfg.t_one_prob > 0:
            t = torch.where(torch.rand(B, device=device) < cfg.t_one_prob, torch.ones_like(t), t)
        t_bc = t[:, None, None, None, None]
        z_t = t_bc * z_src + (1.0 - t_bc) * z_tgt
        v_target = z_src - z_tgt

        x_views, view_ids = self.backbone.pack_views(z_t)
        v_tok, _ = self.backbone.pack_views(v_target)
        pred = self._forward(x_ref, ref_ids, x_views.to(self._dtype), view_ids.to(device), t, cams)

        latent_loss = F.mse_loss(pred.float(), v_tok.float())
        loss = latent_loss
        out = {"latent_loss": latent_loss.detach().item(), "v_mag": v_tok.float().abs().mean().item()}

        if cfg.lpips_loss_weight > 0 or cfg.l1_pixel_loss_weight > 0:
            z0_hat = z_t - t_bc * self.backbone.unpack_views(pred, V, h, w).to(z_t.dtype)
            lpips_l, l1_l = self._pixel_losses(z0_hat, tgt_imgs)
            wt = torch.ones_like(t)
            if cfg.pixel_loss_t_weighting == "one_minus_t":
                wt = 1.0 - t
            elif cfg.pixel_loss_t_weighting == "one_minus_t_noise" and noise_alpha is not None:
                # Noise samples (alpha > 0) only; clean-up samples keep full weight.
                wt = torch.where(noise_alpha.to(device) > 0, 1.0 - t, wt)
            wt = wt.repeat_interleave(V).to(lpips_l.dtype)
            loss = (
                loss
                + cfg.lpips_loss_weight * (wt * lpips_l).mean()
                + cfg.l1_pixel_loss_weight * (wt * l1_l).mean()
            )
            out["lpips_loss"] = lpips_l.mean().detach().item()
            out["l1_pixel_loss"] = l1_l.mean().detach().item()

        out["loss"] = loss
        return out

    def _pixel_losses(self, z0_hat: torch.Tensor, tgt_imgs: torch.Tensor):
        """Per-image LPIPS and L1 between decoded ``z0_hat`` (B, V, C, h, w) and targets."""
        z = z0_hat.flatten(0, 1)
        tgt = tgt_imgs.flatten(0, 1)
        P = self.backbone.PATCH
        h, w = z.shape[-2:]
        H, W = tgt.shape[-2:]
        top, left = (H - h * P) // 2, (W - w * P) // 2
        tgt = tgt[..., top:top + h * P, left:left + w * P]

        crop = self.cfg.pixel_loss_latent_crop
        if crop is not None and (crop < h or crop < w):
            ch, cw = min(crop, h), min(crop, w)
            i = int(torch.randint(0, h - ch + 1, (1,)))
            j = int(torch.randint(0, w - cw + 1, (1,)))
            z = z[..., i:i + ch, j:j + cw]
            tgt = tgt[..., i * P:(i + ch) * P, j * P:(j + cw) * P]

        # One image at a time, recomputed in backward: the full-resolution decoder
        # activations of every view would otherwise stay alive until the loss backward.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = torch.cat([
                checkpoint(self.ae.decode, zi, use_reentrant=False) if torch.is_grad_enabled() else self.ae.decode(zi)
                for zi in z.float().split(1)
            ])
        pred, tgt = pred.float(), tgt.float()
        l1_l = (pred - tgt).abs().mean(dim=(1, 2, 3))
        lpips_l = self.lpips(pred, tgt).flatten() if self.lpips is not None else torch.zeros_like(l1_l)
        return lpips_l, l1_l

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        src_imgs: torch.Tensor,
        ref_imgs: torch.Tensor,
        num_steps: int = 1,
        camera_c2w: Optional[torch.Tensor] = None,
        camera_fxfycxcy: Optional[torch.Tensor] = None,
        noise_alpha: Optional[torch.Tensor] = None,
        tgt_imgs: Optional[torch.Tensor] = None,
        camera_model: str = "pinhole",
    ) -> torch.Tensor:
        """Euler from t=1 to t=0. Returns ``(B, V, 3, H', W')`` in [-1, 1].

        With noise_space=latent, samples with ``noise_alpha=1`` start from pure
        Gaussian noise (``src_imgs`` is ignored: pure NVS). ``0 < alpha < 1``
        also needs ``tgt_imgs``.
        """
        B, V = src_imgs.shape[:2]
        device = src_imgs.device
        z_src, _, x_ref, ref_ids, cams, (h, w) = self._prepare(
            src_imgs, ref_imgs, camera_c2w, camera_fxfycxcy, noise_alpha, tgt_imgs, camera_model
        )
        x, view_ids = self.backbone.pack_views(z_src)
        x, view_ids = x.to(self._dtype), view_ids.to(device)

        num_steps = max(1, int(num_steps))
        ts = [1.0 - i / num_steps for i in range(num_steps + 1)]
        for t_curr, t_next in zip(ts[:-1], ts[1:]):
            t_vec = torch.full((B,), t_curr, device=device, dtype=self._dtype)
            x = x + (t_next - t_curr) * self._forward(x_ref, ref_ids, x, view_ids, t_vec, cams)

        z_out = self.backbone.unpack_views(x, V, h, w).flatten(0, 1)
        imgs = self.decode(z_out)
        return imgs.reshape(B, V, *imgs.shape[1:])
