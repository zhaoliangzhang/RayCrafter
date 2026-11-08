"""Full matched four-dataset evaluation of scheduled concat + UniRoPE 100k."""
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import os
import sys

os.environ.setdefault('CAMERA_VARIANT','unirope')
assert os.environ['CAMERA_VARIANT']=='unirope'
CODE=Path(__file__).resolve().parent
TRAIN_CODE=CODE/'training_snapshot'
REPO=TRAIN_CODE/'code'
ENGINE=CODE/'engine'
ROOT=Path('/cis/net/io81/data/xliu253/omniraycrafter')
VARIANT='unirope'
MODEL='concat_scheduled_unirope'
EXPECTED_STEP=100000
EXPECTED_KEYS=882
COMPONENTS=('unirope_depth',)
CHECKPOINT=ROOT/'mixed_concat_scheduled_unirope_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io105/step-100000.safetensors'
OUT=CHECKPOINT.parent/'evaluations/step100000_hm3d_replica_jhu_dl3dv_q3_20261105'
BASE=OUT
MATCHED=ROOT/'mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104/evaluations/step50000_dl3dv_hm3d_replica_jhu_q3_20261028'
COMBINED=ROOT/'mixed_concat_scope_ff_unirope_v3_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io105/evaluations/step50000_hm3d_dl3dv_jhu_replica_q3_20261104'
PREVIOUS=Path('/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/evaluations/step650000_hm3d_replica_jhu_q3_256x512_20260917')
TOOLS=CODE.parent/'panorama_nvs_transfer_20260818/workspace/baselines/replica_2ref3'
HOST='io95'
WAN=Path('/export/io95/data/xliu253/framecrafter_wan21')
CACHE_ROOT=Path('/export/io95/data/xliu253/scheduled_unirope_eval_100000_20261105')
CACHE=CACHE_ROOT
DREAM_SOURCE=Path('/export/io95/data/xliu253/concat_scopeff_unirope_eval_22000_20261103/dreamsim')
DREAM_CACHE=DREAM_SOURCE
FID_SOURCE=Path('/export/io95/data/xliu253/concat_scopeff_unirope_eval_22000_20261103/fid/inception-2015-12-05.pt')
FID_PACKAGES=ROOT/'evaluations/hm3d_q3_scope48000_replica48000_256x512_20261020/runtime/fid_packages'
PYTHON=Path('/cis/home/xliu253/micromamba/envs/framecrafter/bin/python')
METRIC_PYTHON=PREVIOUS/'runtime/dreamsim_env/bin/python'
COUNTS={'HM3D':423,'Replica':68,'JHU':500,'DL3DV':2800}
TASKS=('nvs','cleanup')
METRICS=('psnr','ssim','lpips','dreamsim')
BASE_SEED=20260904
VRAM_LIMIT_GIB=32
GPU_INDICES=list(range(8))
GPUS=['GPU-fcfa9a32-08ec-b6bc-514c-e94d248b4f19','GPU-8b4c39b3-d294-acf1-a3e0-f4e3d45549ea',
      'GPU-d01abde5-f58a-98df-e2af-b27f390a6666','GPU-d97089ea-d914-5b89-ac07-12acaee2e2fb',
      'GPU-57a353c7-a517-307d-22a7-44ab0ed9e876','GPU-78f722f1-1433-ead7-4912-fe329a0c654a',
      'GPU-6972292e-925b-5de2-83c4-e5b6d2e028fe','GPU-b75ff6d9-e49d-a797-cd09-ee59de580966']
NUM_SHARDS=8

def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if immutable and path.exists():
        assert path.read_text()==text,f'Existing protocol differs: {path}'
        return
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
def link(source,destination):
    source,destination=Path(source).absolute(),Path(destination)
    assert source.exists(),source
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():assert destination.resolve()==source.resolve(),destination
    elif destination.exists():raise FileExistsError(destination)
    else:destination.symlink_to(source,target_is_directory=source.is_dir())
def atomic_png(image,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');image.save(tmp,format='PNG');tmp.replace(path)
def imports():sys.path[:0]=[str(TRAIN_CODE),str(REPO),str(TOOLS)]
def manifest():return json.loads((OUT/'benchmark/manifest.json').read_text())
def bundle_at(root,row,task):
    scene=row['scene_name'].split('::')[-1]
    assert '/' not in scene and scene not in ('.','..')
    assert '/' not in row['id'] and row['id'] not in ('.','..')
    return root/'bundles'/row['dataset']/task/scene/row['id']
def bundle(row,task):return bundle_at(OUT,row,task)
def prediction(row,task,j):return bundle(row,task)/'predictions'/f'query_{j}.png'
def completion(row,task):return bundle(row,task)/'_SUCCESS'
def completed(row,task):
    if not completion(row,task).is_file():return False
    r=json.loads(completion(row,task).read_text())
    assert (r['set_id'],r['task'],r['seed'])==(row['id'],task,row['seed'])
    assert (r['width'],r['height'])==(row['width'],row['height'])
    assert r['diffusion_steps']==20 and not r['target_rgb_accessed_during_inference']
    assert r['model']==MODEL and r['injection_weight']==1.
    assert r['checkpoint_sha256']==json.loads((OUT/'provenance.json').read_text())['checkpoint_sha256']
    return all(prediction(row,task,j).is_file() for j in range(3))
def code_hashes():
    paths=[p for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml','.json') and '__pycache__' not in p.parts]
    paths.append(TOOLS/'score_predictions.py')
    return {str(p):sha256(p) for p in sorted(paths)}
def smoke_rows(rows):return [next(r for r in rows if r['dataset']==d) for d in COUNTS]
@lru_cache(None)
def near_depth_table():return json.loads((OUT/'benchmark/near_depth_resolved.json').read_text())
def near_depth_for(row):return near_depth_table()[row['scene_name']]['near_depth']
def visual_links(row,task):
    b=bundle(row,task);(b/'predictions').mkdir(parents=True,exist_ok=True)
    for j,r in enumerate(row['references']):link(r['image'],b/'references'/f'ref_{j}.png')
    for j,t in enumerate(row['targets']):
        link(t['metric_gt'],b/'GT'/f'query_{j}.png');link(t['image'],b/'GT_original'/f'query_{j}.png')
        if task=='cleanup':link(t['noisy'],b/'noisy_inputs'/f'query_{j}.png')
    for name in ('references','GT','GT_original'):link(b/name,b/'predictions'/name)
    if task=='cleanup':
        link(row['prior_metadata'],b/'noisy_inputs/provenance.json');link(b/'noisy_inputs',b/'predictions/noisy_inputs')
    return b
