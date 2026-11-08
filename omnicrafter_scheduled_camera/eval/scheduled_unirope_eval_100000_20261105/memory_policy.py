"""Device placement only; keep camera arithmetic/checkpoint weights unchanged."""


def restore_camera_modules(dit, device):
    count = 0
    for block in dit.blocks:
        for name in ('scope_pe', 'rayrope_depth', 'unirope_depth'):
            if hasattr(block.self_attn, name):
                getattr(block.self_attn, name).to(device)
                count += 1
    return count


def retain_injected_camera_devices(pipe, device):
    """The parent DiTBlock.onload() recursively moves new camera modules to CPU.

    These modules were installed AFTER DiffSynth wrapped the backbone, so they
    have no independent onload hook. Restore only the new camera modules after
    parent onloading and before denoising. Large pretrained layers retain the
    unmodified bounded memory-management policy.
    """
    original = pipe.load_models_to_device
    def load(model_names):
        result = original(model_names)
        if 'dit' in model_names:
            restore_camera_modules(pipe.dit, device)
        return result
    pipe.load_models_to_device = load
