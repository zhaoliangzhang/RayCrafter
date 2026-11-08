"""OmniCrafter mixed manifest; preserve the existing task mixture and images.

One Bernoulli draw per sample selects all-three-black NVS versus cleanup.
Cleanup redraws augmentation if it accidentally erases all three renders;
this prevents the cleanup branch from silently adding extra all-black sets.
No manifest is rewritten. Near-depth estimates must already be attached.
"""
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from camera import ordered_cameras


def query_dropout(image, rng, low=0, high=100):
    ratio = rng.randint(low, high) / 100.
    pixels = np.asarray(image, dtype=np.uint8).copy()
    remove = np.random.default_rng(rng.getrandbits(64)).random(pixels.shape[:2]) < ratio
    pixels[remove] = 0
    return Image.fromarray(pixels), ratio


def condition_queries(images, rng, nvs_probability=.5, *, task=None):
    if len(images) != 3 or not 0 <= nvs_probability <= 1:
        raise ValueError('Expected three renders and a valid sample-level probability')
    if task is None:
        task = 'nvs' if rng.random() < nvs_probability else 'cleanup'
    if task not in ('nvs', 'cleanup'):
        raise ValueError('Unknown conditioning task')
    if task == 'nvs':
        return [Image.new('RGB', im.size) for im in images], [1.] * 3, 'nvs', 0
    if not any(im.getbbox() is not None for im in images):
        raise ValueError('Cleanup sample has three empty source renders; fix the manifest, not the task label')
    for redraws in range(32):
        noisy, ratios = map(list, zip(*(query_dropout(im, rng) for im in images)))
        if any(im.getbbox() is not None for im in noisy):
            return noisy, ratios, 'cleanup', redraws
    # A very sparse source can be erased even at <100% dropout. Retain one
    # original nonempty render rather than silently turning cleanup into NVS.
    keep = next(i for i, im in enumerate(images) if im.getbbox() is not None)
    noisy[keep], ratios[keep] = images[keep].copy(), 0.
    return noisy, ratios, 'cleanup', 32


class MixedProjectionDataset(Dataset):
    def __init__(self, manifest, offsets, nvs_probability=.5, path_map=None):
        if not 0 <= nvs_probability <= 1:
            raise ValueError('nvs_probability must be in [0,1]')
        self.manifest = Path(manifest)
        self.offsets = np.load(offsets, mmap_mode='r')
        self.nvs_probability = nvs_probability
        self.path_map = path_map or {}
        self._handle, self._pid = None, None
        self._cleanup_groups = None
        self._empty_render_indices = set()

    @staticmethod
    def group_key(row):
        return (row['dataset'], row['projection'], row['height'], row['width'])

    def prepare_cleanup_groups(self):
        """Metadata-only index; never removes rows or edits the manifest."""
        if self._cleanup_groups is not None:
            return
        datasets, scenes = {}, {}
        count = 0
        with self.manifest.open('rb') as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['split'] != 'train':
                    raise ValueError('Never include validation/test rows in cleanup replacements')
                key = self.group_key(row)
                datasets.setdefault(key, []).append(index)
                scenes.setdefault((*key, row['scene_name']), []).append(index)
                count += 1
        if count != len(self):
            raise ValueError('Manifest and offsets lengths differ')
        self._cleanup_groups = datasets, scenes

    def source_renders(self, row):
        h, w = row['height'], row['width']
        if (row['projection'], h, w) not in (('panorama', 256, 512), ('perspective', 288, 512)):
            raise ValueError('Mixed training requires panorama HxW=256x512 / perspective HxW=288x512')
        if len(row['targets']) != 3:
            raise ValueError('Expected three target renders')
        return [self.load_image(v['noisy'], h, w) for v in row['targets']]

    def replacement_for_cleanup(self, original, index):
        """Keep the chosen task/dataset; replace the WHOLE aligned sample."""
        self.prepare_cleanup_groups()
        datasets, scenes = self._cleanup_groups
        key = self.group_key(original)
        self._empty_render_indices.add(index)
        seen = {index}
        attempts = 0
        pools = [(scenes[(*key, original['scene_name'])], 64), (datasets[key], 256)]
        for pool, limit in pools:
            for candidate in random.sample(pool, min(limit, len(pool))):
                if candidate in seen or candidate in self._empty_render_indices:
                    continue
                seen.add(candidate)
                row = self.record(candidate)
                if self.group_key(row) != key:
                    raise RuntimeError('Replacement would change dataset, projection or resolution')
                renders = self.source_renders(row)
                attempts += 1
                if any(im.getbbox() is not None for im in renders):
                    print(json.dumps({'event': 'cleanup_source_resampled',
                        'rank': int(os.environ.get('RANK', '0')), 'dataset': row['dataset'],
                        'original_index': index, 'original_id': original['id'],
                        'original_scene': original['scene_name'], 'replacement_index': candidate,
                        'replacement_id': row['id'], 'replacement_scene': row['scene_name'],
                        'attempts': attempts}), flush=True)
                    return row, renders, candidate, attempts
                self._empty_render_indices.add(candidate)
        raise RuntimeError(f'No nonempty cleanup source found in bounded same-dataset search: '
                           f'dataset={original["dataset"]} sample={original["id"]}')

    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle, self._pid = self.manifest.open('rb'), os.getpid()
        self._handle.seek(int(self.offsets[index]))
        record = json.loads(self._handle.readline())
        if record['split'] != 'train':
            raise ValueError('Never train on validation/test rows')
        return record

    def load_image(self, path, h, w):
        for old, new in self.path_map.items():
            if path.startswith(old.rstrip('/') + '/'):
                path = new.rstrip('/') + path[len(old.rstrip('/')):]
                break
        with Image.open(path) as image:
            return image.convert('RGB').resize((w, h), Image.Resampling.BILINEAR)

    def __getitem__(self, index):
        original = row = self.record(index)
        source_renders = self.source_renders(row)
        # Draw the task exactly ONCE, before any replacement search. In
        # particular, never convert an unusable cleanup draw into extra NVS.
        task = 'nvs' if random.random() < self.nvs_probability else 'cleanup'
        actual_index, attempts = index, 0
        if task == 'cleanup' and not any(im.getbbox() is not None for im in source_renders):
            row, source_renders, actual_index, attempts = self.replacement_for_cleanup(row, index)
        # References, targets, camera poses, near-depth and renders must ALL
        # belong to the replacement row; do not mix geometry across samples.
        row, cameras = ordered_cameras(row)
        h, w = row['height'], row['width']
        if (row['projection'], h, w) not in (('panorama', 256, 512), ('perspective', 288, 512)):
            raise ValueError('Mixed training requires panorama HxW=256x512 / perspective HxW=288x512')
        load = lambda path: self.load_image(path, h, w)
        refs = [load(v['image']) for v in row['references']]
        clean = [load(v['image']) for v in row['targets']]
        noisy, ratios, task, redraws = condition_queries(source_renders, random, self.nvs_probability, task=task)
        all_black = all(im.getbbox() is None for im in noisy)
        if all_black != (task == 'nvs'):
            raise RuntimeError('Per-sample task/image invariant violated')
        return {'input_images': refs, 'conditioning_images': noisy, 'target_images': refs + clean,
                'prompt': '', 'sample_id': row['id'], 'scene_name': row['scene_name'],
                'dataset': row['dataset'], 'projection': row['projection'],
                'sample_task_type': task, 'all_queries_black': all_black,
                'cleanup_dropout_redraws': redraws,
                'original_sample_id': original['id'], 'sample_index': actual_index,
                'cleanup_source_resampled': actual_index != index,
                'cleanup_scene_changed': row['scene_name'] != original['scene_name'],
                'cleanup_resample_attempts': attempts,
                'query_pixel_dropout_ratios': ratios, 'camera_inputs': cameras}

    def __getstate__(self):
        return dict(self.__dict__, _handle=None, _pid=None)


def collate(rows):
    return rows
