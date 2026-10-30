"""Keep exact constant conditioning without keeping unused frozen encoders.

Run before accelerator.prepare while the DiT and VAE are still on CPU. Only
the frozen T5 temporarily visits the rank's GPU. The cached bf16 context is a
non-persistent pipeline buffer: it follows pipeline.to(), but never enters an
adapter checkpoint. The original individual-frame CLIP unit is already a no-op.
"""
import gc
import sys
import torch
from common import REPO, FROZEN_ENCODER_POLICY
sys.path.insert(0, str(REPO))
from diffsynth.pipelines.wan_video import (
    WanVideoUnit_PromptEmbedder, WanVideoUnit_ImageEmbedderCLIP,
)


class CachedEmptyPrompt(WanVideoUnit_PromptEmbedder):
    def __init__(self):
        super().__init__()
        self.onload_model_names = ()

    def process(self, pipe, prompt, positive):
        if prompt != '':
            raise ValueError('Cached training context supports only the empty prompt')
        assert pipe.text_encoder is None
        assert pipe.empty_prompt_context.device == torch.device(pipe.device)
        return {'context': pipe.empty_prompt_context}


@torch.no_grad()
def release_frozen_encoders(pipe, device):
    """Cache the original GPU T5 result, check determinism, then release T5/CLIP."""
    device = torch.device(device)
    assert pipe.dit.individual_encoding, 'CLIP is unused only in this path'
    assert not hasattr(pipe, 'empty_prompt_context'), 'Already prepared'
    units = [u for u in pipe.units if type(u) is WanVideoUnit_PromptEmbedder]
    assert len(units) == 1
    assert all(p.device.type == 'cpu' for p in pipe.dit.parameters())
    byte_counts = {}
    for name in ('text_encoder', 'image_encoder'):
        encoder = getattr(pipe, name)
        assert encoder is not None and not any(p.requires_grad for p in encoder.parameters())
        byte_counts[name] = sum(t.numel()*t.element_size() for t in encoder.parameters())
        encoder.eval()
    original_device = pipe.device
    pipe.text_encoder.to(device=device, dtype=pipe.torch_dtype)
    pipe.device = device
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=rng_devices):
            first = units[0].encode_prompt(pipe, '')
            repeated = units[0].encode_prompt(pipe, '')
        assert first.shape == repeated.shape and first.ndim == 3 and first.shape[0] == 1
        assert torch.isfinite(first).all() and torch.equal(first, repeated), 'Frozen empty prompt was not deterministic'
        cached = first.detach().cpu().clone()
    finally:
        pipe.device = original_device
    pipe.register_buffer('empty_prompt_context', cached, persistent=False)
    pipe.text_encoder = None
    pipe.image_encoder = None
    pipe.units = [CachedEmptyPrompt() if type(u) is WanVideoUnit_PromptEmbedder else u
                  for u in pipe.units if type(u) is not WanVideoUnit_ImageEmbedderCLIP]
    del first, repeated, encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    assert not any(n.startswith(('text_encoder.', 'image_encoder.')) for n, _ in pipe.named_parameters())
    return {'policy': FROZEN_ENCODER_POLICY, 'encoder_parameter_bytes_removed': byte_counts,
            'context_shape': list(cached.shape), 'context_bytes': cached.numel()*cached.element_size(),
            'context_dtype': str(cached.dtype), 'computed_on': str(device),
            'repeat_embedding_bit_exact': True, 'vae_device_during_training': 'cuda',
            'optimizer_offload': False}
