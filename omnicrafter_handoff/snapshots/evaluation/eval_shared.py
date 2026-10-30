"""Frozen mixed-projection evaluation protocol; never changes training files."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import sys

CODE = Path(__file__).resolve().parent
SNAPSHOT = CODE/'training_snapshot'
REPO = SNAPSHOT/'code'
TRAIN = Path('/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104')
CHECKPOINT = TRAIN/'step-50000.safetensors'
OUT = TRAIN/'evaluations/step50000_dl3dv_hm3d_replica_jhu_q3_20261028'
MATCHED = TRAIN/'evaluations/step22000_dl3dv_hm3d_replica_jhu_q3_20261024'
ROOT = CODE.parent/'panorama_nvs_transfer_20260818/workspace'
TOOLS = ROOT/'baselines/replica_2ref3'
PREVIOUS = Path('/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/evaluations/step650000_hm3d_replica_jhu_q3_256x512_20260917')
PRIORS = Path('/cis/net/io81/data/xliu253/framecrafter/evaluations/request_20260904/panovggt/q3_grouped')
PERSPECTIVE = Path('/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test')
WAN = Path('/export/io95/data/xliu253/framecrafter_wan21')
CACHE = Path('/export/io95/data/xliu253/mixed_projection_eval_50000_20261028')
DREAM_CACHE = Path('/export/io95/data/xliu253/oc_teacher_student_benchmark_cache/dreamsim')
FID_SOURCE = Path('/cis/net/io96/data/xliu253/oc650k_eval_cache/fid/inception-2015-12-05.pt')
FID_PACKAGES = Path('/cis/net/io81/data/xliu253/omniraycrafter/evaluations/hm3d_q3_scope48000_replica48000_256x512_20261020/runtime/fid_packages')
PYTHON = Path('/cis/home/xliu253/micromamba/envs/framecrafter/bin/python')
METRIC_PYTHON = PREVIOUS/'runtime/dreamsim_env/bin/python'
COUNTS = {'HM3D':423,'Replica':68,'JHU':500,'DL3DV':2800}
TASKS = ('nvs','cleanup')
METRICS = ('psnr','ssim','lpips','dreamsim')
BASE_SEED = 20260904
GPUS = ['GPU-fcfa9a32-08ec-b6bc-514c-e94d248b4f19','GPU-8b4c39b3-d294-acf1-a3e0-f4e3d45549ea',
        'GPU-d01abde5-f58a-98df-e2af-b27f390a6666','GPU-d97089ea-d914-5b89-ac07-12acaee2e2fb',
        'GPU-57a353c7-a517-307d-22a7-44ab0ed9e876','GPU-78f722f1-1433-ead7-4912-fe329a0c654a',
        'GPU-6972292e-925b-5de2-83c4-e5b6d2e028fe']
NUM_SHARDS = len(GPUS)

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def write_json(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if immutable and path.exists():
        assert path.read_text()==text,f'Immutable protocol differs: {path}'
        return
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text);temporary.replace(path)

def link(source,destination):
    source,destination=Path(source).absolute(),Path(destination)
    assert source.exists(),source
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve()==source.resolve(),destination
    elif destination.exists():
        raise FileExistsError(destination)
    else:
        destination.symlink_to(source,target_is_directory=source.is_dir())

def atomic_png(image,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    image.save(temporary,format='PNG');temporary.replace(path)

def imports():
    sys.path[:0]=[str(SNAPSHOT),str(REPO),str(TOOLS)]

def manifest():
    return json.loads((OUT/'benchmark/manifest.json').read_text())

def bundle(row,task):
    scene=row['scene_name'].split('::')[-1]
    assert '/' not in scene and scene not in ('.','..')
    return OUT/'bundles'/row['dataset']/task/scene/row['id']

def prediction(row,task,number):
    return bundle(row,task)/'predictions'/f'query_{number}.png'

def completion(row,task):
    return bundle(row,task)/'_SUCCESS'

def completed(row,task):
    if not completion(row,task).is_file():
        return False
    record=json.loads(completion(row,task).read_text())
    assert record['set_id']==row['id'] and record['task']==task and record['seed']==row['seed']
    assert record['width']==row['width'] and record['height']==row['height']
    assert record['diffusion_steps']==20 and not record['target_rgb_accessed_during_inference']
    return all(prediction(row,task,j).is_file() for j in range(3))

def code_hashes():
    paths=[p for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml') and '__pycache__' not in p.parts]
    paths += [TOOLS/'score_predictions.py']
    return {str(p.relative_to(CODE)) if p.is_relative_to(CODE) else str(p):sha256(p) for p in sorted(paths)}

def smoke_rows(rows):
    # Exercise both resolutions and every dataset without changing any groups.
    return [next(row for row in rows if row['dataset']==dataset) for dataset in COUNTS]
