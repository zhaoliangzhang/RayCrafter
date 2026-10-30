"""Small, dependency-light helpers for the cluster handoff (no SSH or jobs)."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct

ROOT = Path(__file__).resolve().parents[1]
IMAGE_KEYS = ('image', 'noisy', 'metric_gt')
COUNTS = {'DL3DV': 100000, 'HM3D': 60000, 'JHU': 40000}
EVAL_COUNTS = {'HM3D': 423, 'Replica': 68, 'JHU': 500, 'DL3DV': 2800}
PUBLISHED_SHA256 = 'fa22ffdb9a54d5eb029618bcd424af1623ee6d64eb6277db7f7c478ab5b92524'


def camera_architecture(config):
    """Old configs retain concat; SCoPE must be selected explicitly."""
    value = config.get('architecture', 'concat')
    if value not in ('concat', 'scope'):
        raise ValueError('architecture must be concat or scope')
    if 'camera_injection' in config and config['camera_injection'] != value:
        raise ValueError('Use architecture, not a conflicting camera_injection field')
    return value


def scope_parameter_shapes():
    """The exact Q/K-only, linear projection, width-5120 WAN SCoPE variant."""
    suffixes = {'eq.weight': [5120, 7], 'ek.weight': [5120, 7],
        'norm_pe_q.weight': [5120], 'norm_pe_k.weight': [5120],
        'alpha_q': [1], 'alpha_k': [1], 'scale_gate.0.weight': [1280, 1],
        'scale_gate.0.bias': [1280], 'scale_gate.2.weight': [5120, 1280],
        'scale_gate.2.bias': [5120]}
    return {f'blocks.{i}.self_attn.scope_pe.{suffix}': shape
            for i in range(40) for suffix, shape in suffixes.items()}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def text_open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()


def image_fields(row, metadata=True):
    for view in row['references'] + row['targets']:
        for key in IMAGE_KEYS:
            if key in view:
                yield view, key
    if metadata and row.get('prior_metadata'):
        yield row, 'prior_metadata'


def safe_absolute(value):
    p = PurePosixPath(value)
    if not p.is_absolute() or '..' in p.parts or '\x00' in value or '\n' in value:
        raise ValueError(f'Unsafe/non-absolute path: {value!r}')
    return p


def rebase(value, mapping):
    p = safe_absolute(value)
    for old, new in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        old, new = safe_absolute(old), safe_absolute(new)
        if p.is_relative_to(old):
            return str(new / p.relative_to(old))
    raise ValueError(f'No prefix mapping for required asset: {value}')


def validate_row(row, train=False):
    if len(row['references']) != 2 or len(row['targets']) != 3:
        raise ValueError('This handoff entry point requires 2ref3')
    expected = {'panorama': (256, 512), 'perspective': (288, 512)}
    if expected.get(row['projection']) != (row['height'], row['width']):
        raise ValueError('Projection/resolution mismatch (height, width)')
    if train and row.get('split') != 'train':
        raise ValueError('Validation/test row in training manifest')
    if [t['target_number'] for t in row['targets']] != [0, 1, 2]:
        raise ValueError('Target order must be exactly 0,1,2')
    for view in row['references'] + row['targets']:
        pose = view['w2c']
        if len(pose) != 4 or any(len(r) != 4 for r in pose):
            raise ValueError('Expected 4x4 w2c poses')
        if row['projection'] == 'perspective' and 'camera' not in view:
            raise ValueError('Perspective views require calibration and distortion')
    for view, key in image_fields(row, metadata=False):
        safe_absolute(view[key])


def checkpoint_header(path, architecture='concat', *, initialization=False):
    camera_architecture({'architecture': architecture})
    with Path(path).open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Incomplete safetensors header')
        n = struct.unpack('<Q', raw)[0]
        if not 0 < n < 100 * 1024**2:
            raise ValueError('Unreasonable safetensors header size')
        data = json.loads(f.read(n))
    channels = 36 if architecture == 'scope' else 420
    if data.get('patch_embedding.weight', {}).get('shape') != [5120, channels, 1, 2, 2]:
        raise ValueError(f'Expected {architecture} patch embedding with {channels} input channels')
    keys = set(data) - {'__metadata__'}
    expected_scope = scope_parameter_shapes() if architecture == 'scope' and not initialization else {}
    if len(keys) != 802 + len(expected_scope):
        raise ValueError(f'Expected {802 + len(expected_scope)} tensors for {architecture}')
    if {k for k in keys if '.scope_pe.' in k} != set(expected_scope):
        raise ValueError('SCoPE Q/K-only parameter names do not match this implementation')
    for key, shape in expected_scope.items():
        if data[key]['shape'] != shape:
            raise ValueError(f'Wrong SCoPE parameter shape: {key}')
    declared = data.get('__metadata__', {}).get('camera_injection')
    if declared is not None and declared != architecture:
        raise ValueError('Checkpoint camera_injection metadata disagrees with architecture')
    return data.get('__metadata__', {})


def required_file(value):
    p = Path(value).expanduser().resolve(strict=True)
    if not p.is_file():
        raise ValueError(f'Expected file: {p}')
    return p


def gpu_tokens(number):
    """Respect scheduler CUDA_VISIBLE_DEVICES; never discover/claim other GPUs."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = visible.split(',') if visible is not None else [str(i) for i in range(number)]
    if number < 1 or len(tokens) < number or any(not t.strip() or t.strip() == '-1' for t in tokens):
        raise ValueError('Insufficient GPUs in CUDA_VISIBLE_DEVICES')
    return [t.strip() for t in tokens[:number]]


def protocol_hashes():
    dirs = [ROOT/'tools', ROOT/'snapshots/evaluation', ROOT/'snapshots/mixed']
    return {str(p.relative_to(ROOT)): sha256(p) for directory in dirs
            for p in sorted(directory.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.sh')}
