"""Lazy one-logical-set-per-line loader with projection-specific image geometry."""
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode
from common import DATA, REPO
sys.path.insert(0, str(REPO))
from diffsynth.core.data.pose_conditioned_render_query_dataset import WanPanoramaPoseConditionedRenderQueryDataset
from mixed_camera import camera_features
from runtime_guard import event


class MixedProjectionDataset(Dataset):
    def __init__(self, architecture, manifest=DATA/'train.jsonl', offsets=DATA/'train.offsets.npy', nvs_probability=.5):
        if architecture not in ('scope','concat') or not 0 <= nvs_probability <= 1:
            raise ValueError('Invalid architecture or task mixture')
        self.architecture = architecture
        self.manifest = Path(manifest)
        self.offsets = np.load(offsets,mmap_mode='r')
        self.nvs_probability = nvs_probability
        self.min_pixel_dropout,self.max_pixel_dropout = 0.,1.
        self._handle,self._pid = None,None

    def __len__(self):
        return len(self.offsets)

    def record(self,index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle,self._pid = self.manifest.open('rb'),os.getpid()
        self._handle.seek(int(self.offsets[index]))
        record = json.loads(self._handle.readline())
        assert record['split'] == 'train'
        assert len(record['references']) == 2 and len(record['targets']) == 3
        return record

    @staticmethod
    def load_image(path,height,width):
        event('image_read',path=str(path))
        with Image.open(path) as image:
            image = image.convert('RGB')
            if image.size != (width,height):
                image = TF.resize(image,[height,width],interpolation=InterpolationMode.BILINEAR)
            result = image.copy()
        event('image_read_done',path=str(path))
        return result

    def __getitem__(self,index):
        event('manifest_read',dataset_index=index)
        record = self.record(index)
        event('sample_load',sample_id=record['id'])
        h,w = record['height'],record['width']
        load = lambda path:self.load_image(path,h,w)
        refs = [load(v['image']) for v in record['references']]
        clean = [load(v['image']) for v in record['targets']]
        is_nvs = self.nvs_probability > 0 and random.random() < self.nvs_probability
        if is_nvs:
            noisy = [Image.new('RGB',(w,h)) for _ in range(3)]
            ratios = [1.]*3
        else:
            noisy,ratios = [],[]
            for view in record['targets']:
                image,ratio = WanPanoramaPoseConditionedRenderQueryDataset._degrade_query(self,load(view['noisy']),random)
                noisy.append(image)
                ratios.append(ratio)
        event('camera_features')
        features,scale = camera_features(record,self.architecture)
        task = 'nvs' if is_nvs else 'cleanup'
        return {'input_images':refs,'conditioning_images':noisy,'target_images':refs+clean,
            'prompt':'','sample_id':record['id'],'scene_name':record['scene_name'],
            'dataset':record['dataset'],'projection':record['projection'],
            'slot_roles':['reference']*2+['query']*3,'sample_task_type':task,
            'query_pixel_dropout_ratios':ratios,'target_task_types':[task]*3,
            'query_condition_types':['all_black_nvs' if is_nvs else 'augmented_noisy_render']*3,
            'slot_frame_indices':[v['frame_index'] for v in record['references']+record['targets']],
            'pose_scale':scale,**features}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'],state['_pid'] = None,None
        return state
