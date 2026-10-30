"""Training constants shared by the private mixed-projection code snapshot."""
from settings import *

REPO = CODE / 'code'
BATCH = 1
WORLD_SIZE = 4
GLOBAL_BATCH = BATCH * WORLD_SIZE
FINAL_STEP = 100000
SAVE_STEPS = 2000
GPU_UUIDS = ['GPU-23a20f7a-b46b-89b9-bc05-34adea947a4e',
             'GPU-036f1953-9fc7-7801-1efa-c47563d498e8',
             'GPU-fea0138f-f1e8-08a5-82e6-3bd3c19d8545',
             'GPU-13857956-9ec0-34a2-3751-732032e5e5fb']
WAN = Path('/export/io100/data/xliu253/mixed_neardepth_20261029/wan21')
FROZEN_ENCODER_POLICY = 'cache_empty_t5_once_on_gpu_remove_t5_and_unused_clip_v1'
POSE_NORMALIZATION = 'third_query_frame_per_scene_near_depth_v1'
NEAR_DEPTH_SOURCES = {
    'perspective': Path('/cis/home/zzhan288/RayCrafter/near_depth/dl3dv_points.json'),
    'panorama': Path('/cis/home/zzhan288/RayCrafter/near_depth/framecrafter_5input.json'),
}
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
            and p.suffix in ('.py', '.yaml', '.sh', '.json')}
