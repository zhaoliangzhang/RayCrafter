"""CPU-only published-adapter conversion. Never changes the source checkpoint."""
from handoff_common import *


def image_only_state(state):
    """Preserve every LoRA/bias value and the first 36 patch input channels."""
    if len(state) != 802 or tuple(state['patch_embedding.weight'].shape) != (5120,420,1,2,2):
        raise ValueError('Expected the published 802-tensor, 420-channel FrameCrafter adapter')
    result = dict(state)
    result['patch_embedding.weight'] = state['patch_embedding.weight'][:, :36].contiguous()
    return result


def prepare_scope_initialization(source, destination):
    source, destination = required_file(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to overwrite initialization: {destination}')
    checkpoint_header(source, 'concat')
    source_sha = sha256(source)
    if source_sha != PUBLISHED_SHA256:
        raise ValueError('Only the exact published FrameCrafter adapter can be converted here')
    # Lazy imports keep launcher --dry-run and header tests free of torch/CUDA.
    from safetensors.torch import load_file, save_file
    state = image_only_state(load_file(str(source), device='cpu'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    save_file(state, str(temporary), metadata={
        'source_sha256': source_sha, 'camera_injection': 'scope',
        'kind': 'image-only initialization; not a trained SCoPE checkpoint',
        'conversion': 'keep all LoRA and bias tensors; patch_embedding.weight[:, :36]; initialize SCoPE separately'})
    checkpoint_header(temporary, 'scope', initialization=True)
    # Hard-link creation is atomic and fails if another writer created the target.
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return destination
