"""Training constants shared by the private mixed-projection code snapshot."""
from settings import *

REPO = CODE / 'code'
BATCH = 1
WORLD_SIZE = 4
GLOBAL_BATCH = 4
FINAL_STEP = 300000
SAVE_STEPS = 2000
GPU_UUIDS = ['GPU-de2572d0-76cb-919a-1321-abe91e3a44a2',
             'GPU-0b1d0897-aec2-700e-7f6e-26575c1afb41',
             'GPU-168a3e36-1f09-c2dc-a903-a3b44c4c1399',
             'GPU-e9e3ef51-e821-78af-1246-cfd8821523b7']
WAN = Path('/dev/shm/xliu253_oc650k_uncertainty_8v74nrtf/wan21')
WAN_FALLBACK = Path('/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/checkpoints/wan21')
PUBLISHED = Path('/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/weights/framecrafter_released/framecrafter.safetensors')
PUBLISHED_SHA = 'fa22ffdb9a54d5eb029618bcd424af1623ee6d64eb6277db7f7c478ab5b92524'


def model_paths(wan):
    model = Path(wan) / 'Wan-AI/Wan2.1-I2V-14B-480P'
    shards = sorted(model.glob('diffusion_pytorch_model-*-of-00007.safetensors'))
    assert len(shards) == 7
    components = [model / n for n in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth',
                                    'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    assert all(p.is_file() and p.stat().st_size for p in shards + components)
    tokenizer = Path(wan) / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    assert (tokenizer / 'tokenizer_config.json').is_file()
    return json.dumps([shards] + components, default=str), str(tokenizer)


def config_for_json(value):
    import math
    if isinstance(value, dict):
        return {k: config_for_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [config_for_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def code_hashes():
    return {str(p.relative_to(CODE)):sha256(p) for p in sorted(CODE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.suffix in ('.py', '.yaml', '.sh')}
