"""Launch exactly one authorized four-GPU continuation after queue admission."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from config import ROOT,OUT,LOCAL,INITIALIZATION,FINAL_STEP,RAMP_STEPS,PYTHON,VARIANT
from queue_training import probe

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    report=json.loads((OUT/'preflight.json').read_text())
    for path,expected in report['code_sha256'].items():
        if digest(ROOT/path)!=expected:
            raise RuntimeError(f'Code changed since preflight; refusing stale queued experiment: {path}')
    for path,key in [(LOCAL/'train.jsonl','manifest_sha256'),(LOCAL/'train.offsets.npy','offsets_sha256'),
                     (INITIALIZATION,'initialization_sha256')]:
        if digest(path)!=report[key]:raise RuntimeError(f'Input changed: {path}')
    for projection,path in report['near_depth_source_paths'].items():
        if digest(path)!=report['near_depth_source_sha256'][projection]:
            raise RuntimeError('Near-depth source file changed while queued')
    if not probe()['idle']:
        return 75
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
        TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN',
        TORCHINDUCTOR_CACHE_DIR=str(LOCAL/'inductor'),TRITON_CACHE_DIR=str(LOCAL/'triton'),
        TORCH_EXTENSIONS_DIR=str(LOCAL/'torch_extensions'))
    environment.pop('TORCH_COMPILE_DISABLE',None)
    args=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
        str(ROOT/'train.py'),'--wan',report['wan_root'],'--initialization-checkpoint',str(INITIALIZATION),
        '--manifest',str(LOCAL/'train.jsonl'),'--offsets',str(LOCAL/'train.offsets.npy'),
        '--output',str(OUT),'--max-steps',str(FINAL_STEP),'--save-every','2000',
        '--ramp-steps',str(RAMP_STEPS),'--nvs-probability','0.5','--path-map',str(ROOT/'image_path_map.shared.json')]
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    correctness_log=logs/f'gpu_correctness_{stamp}.log'
    with correctness_log.open('a') as stream:
        check=subprocess.run(args[:8]+[str(ROOT/'gpu_smoke.py')],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    if check.returncode:
        print(f'GPU correctness check failed: {correctness_log}',flush=True)
        return check.returncode
    if not probe()['idle']:
        return 75
    log=logs/f'train_{stamp}.log'
    (OUT/'launch.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
        command=args,variant=VARIANT,log=str(log),gpu_list='0,1,2,3'),indent=2)+'\n')
    print(f'Training log: {log}',flush=True)
    with log.open('a',buffering=1) as stream:
        child=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            stream.write(line);print(line,end='',flush=True)
        return child.wait()

if __name__=='__main__':
    sys.exit(main())
