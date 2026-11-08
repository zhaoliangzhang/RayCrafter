"""CPU-only manifest/near-depth/initialization audit and local metadata cache."""
from collections import Counter
from datetime import datetime
import json
import math
import shutil
from pathlib import Path
import numpy as np
from config import ROOT,VARIANT,HOST,SOURCE,OUT,LOCAL,INITIALIZATION,FINAL_STEP,RAMP_STEPS
from contract import verify_vendor,CAMERA_CONTRACT
from checkpoints import prepare_initialization,sha256
from train import wan_paths
from dataset import MixedProjectionDataset

def source_fingerprints():
    paths = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.txt','.yaml','.json')
             and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}

def choose_wan():
    candidates = [Path('/dev/shm/xliu253_oc650k_uncertainty_8v74nrtf/wan21')] if HOST=='io104' else [Path('/dev/shm/xliu253_mixed_scope_20261028/wan21')]
    candidates.append(Path('/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/checkpoints/wan21'))
    for candidate in candidates:
        try:
            wan_paths(candidate)
        except (FileNotFoundError,PermissionError):
            continue
        return candidate
    raise FileNotFoundError('No complete readable WAN installation')

def main():
    verify_vendor()
    OUT.mkdir(parents=True,exist_ok=True); LOCAL.mkdir(parents=True,exist_ok=True)
    summary=json.loads((SOURCE/'summary.json').read_text())
    expected={'DL3DV':100000,'HM3D':60000,'JHU':40000}
    assert summary['dataset_counts']==expected and summary['validation_used_for_training'] is False
    assert summary['jhu_heldout_scenes_excluded'] is True
    prepare_initialization(INITIALIZATION,OUT/'initialization/concat_step50000_full420.safetensors')
    for name,key in [('train.jsonl','sha256'),('train.offsets.npy','offsets_sha256')]:
        destination=LOCAL/name
        if not destination.exists() or sha256(destination)!=summary[key]:
            temporary=LOCAL/f'.{name}.copying'
            shutil.copyfile(SOURCE/name,temporary)
            assert sha256(temporary)==summary[key]
            temporary.replace(destination)
    depths={}
    for projection,path in summary['near_depth_source_paths'].items():
        assert sha256(path)==summary['near_depth_source_sha256'][projection]
        depths[projection]=json.loads(Path(path).read_text())
        saved=OUT/f'initialization/near_depth_{projection}.json'
        if not saved.exists():
            shutil.copyfile(path,saved)
        assert sha256(saved)==summary['near_depth_source_sha256'][projection]
    offsets=np.load(LOCAL/'train.offsets.npy',mmap_mode='r')
    counts=Counter(); scenes={k:set() for k in expected};picked={};position=0
    with (LOCAL/'train.jsonl').open('rb') as stream:
        for index,line in enumerate(stream):
            assert offsets[index]==position; position+=len(line)
            row=json.loads(line)
            assert row['split']=='train' and len(row['references'])==2 and len(row['targets'])==3
            assert all(v.get('image') and v.get('noisy') for v in row['targets'])
            assert (row['dataset']=='DL3DV')==(row['projection']=='perspective')
            assert (row['height'],row['width'])==((288,512) if row['projection']=='perspective' else (256,512))
            key=row['scene_name'].removeprefix('DL3DV::') if row['projection']=='perspective' else row['scene_name']
            depth=float(depths[row['projection']][key]['near_depth'])
            assert math.isfinite(depth) and depth>0 and row['near_depth']==depth
            counts[row['dataset']]+=1; scenes[row['dataset']].add(row['scene_name'])
            picked.setdefault(row['dataset'],index)
            if (index+1)%50000==0:
                print(f'Audited {index+1} train-only rows, including exact near-depth source values',flush=True)
    assert dict(counts)==expected and len(offsets)==200000
    path_map=json.loads((ROOT/'image_path_map.shared.json').read_text())
    data=MixedProjectionDataset(LOCAL/'train.jsonl',LOCAL/'train.offsets.npy',0.,path_map)
    for dataset,index in picked.items():
        sample=data[index]
        assert sample['sample_task_type']=='cleanup'
        print(f'Decoded 2ref3 GT/noisy images for {dataset}: {sample["sample_id"]}',flush=True)
    wan=choose_wan()
    report=dict(time=datetime.now().astimezone().isoformat(),variant=VARIANT,host=HOST,camera=CAMERA_CONTRACT,
        initialization=str(INITIALIZATION),initialization_sha256=sha256(INITIALIZATION),
        source_manifest=str(SOURCE/'train.jsonl'),local_manifest=str(LOCAL/'train.jsonl'),
        manifest_sha256=summary['sha256'],offsets_sha256=summary['offsets_sha256'],
        near_depth_source_paths=summary['near_depth_source_paths'],near_depth_source_sha256=summary['near_depth_source_sha256'],
        all_rows_near_depth_verified=True,train_only=True,jhu_heldout_excluded=True,
        dataset_counts=dict(counts),scene_counts={k:len(v) for k,v in scenes.items()},
        wan_root=str(wan),final_step=FINAL_STEP,additional_steps=300000,ramp_steps=RAMP_STEPS,
        batch_per_gpu=1,global_batch=4,nvs_probability=.5,task_sampling='per_sample_all_three_black',
        zero_stage=2,optimizer_device='GPU',cpu_offload=False,code_sha256=source_fingerprints())
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='code_sha256'},indent=2),flush=True)

if __name__=='__main__':
    main()
