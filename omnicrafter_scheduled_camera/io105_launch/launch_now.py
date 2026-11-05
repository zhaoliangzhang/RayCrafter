"""User-requested immediate UniRoPE launch sharing io105 with their old job.

The pinned training code stays unchanged. Only the admission policy changes:
allow the four verified existing xliu253 training PIDs, require >=64 GiB free
per GPU, and reject any other process. Never stop any existing training PID.
"""
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/cis/home/xliu253/Documents/PanoLagerNVS/omnicrafter_scheduled_camera_20261104')
os.environ['CAMERA_VARIANT']='unirope'
sys.path.insert(0,str(ROOT))
from config import OUT,LOCAL,INITIALIZATION,FINAL_STEP,RAMP_STEPS,PYTHON,GPU_UUIDS
from queue_training import probe,status,Tee
from run_job import digest

OLD_OUTPUT='/cis/net/io81/data/xliu253/omniraycrafter/mixed_concat_scope_ff_unirope_v3_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io105'
ALLOWED=dict(zip(GPU_UUIDS,[663291,663292,663293,663294]))

def check_admission():
    result=probe()  # Checks the exact four authorized GPU UUIDs.
    for process in result['compute_processes']:
        pid=process['pid']
        if ALLOWED[process['uuid']]!=pid:
            raise RuntimeError(f'Unexpected GPU process; will not share: {process}')
        proc=Path('/proc')/str(pid)
        if proc.stat().st_uid!=os.getuid():
            raise RuntimeError('Existing GPU process is not owned by this user')
        args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'train.py --wan ' not in args or OLD_OUTPUT not in args:
            raise RuntimeError('Existing PID no longer identifies the verified old training')
    text=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=30).stdout
    free={r[0]:int(r[1]) for r in csv.reader(text.strip().splitlines(),skipinitialspace=True)}
    if set(free)!=set(GPU_UUIDS) or min(free.values())<64*1024:
        raise RuntimeError(f'Need at least 64 GiB free per card before starting: {free}')
    return dict(result,free_mib=free,user_requested_shared_launch=True,existing_training_untouched=True)

def run_logged(args,environment,path):
    with path.open('a',buffering=1) as stream:
        child=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            stream.write(line);print(line,end='',flush=True)
        return child.wait()

def main():
    if list(OUT.glob('step-*.safetensors')):
        raise RuntimeError('Refusing implicit restart of an already-trained experiment')
    with (OUT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        logs=OUT/'logs';logs.mkdir(exist_ok=True)
        controller_log=(logs/f'immediate_launch_{stamp}.log').open('a',buffering=1)
        sys.stdout=Tee(sys.stdout,controller_log)
        admission=check_admission()
        status('validating_user_requested_shared_launch',**admission)
        report=json.loads((OUT/'preflight.json').read_text())
        for path,expected in report['code_sha256'].items():
            if digest(ROOT/path)!=expected:raise RuntimeError(f'Pinned code changed: {path}')
        for path,key in [(LOCAL/'train.jsonl','manifest_sha256'),(LOCAL/'train.offsets.npy','offsets_sha256'),
                         (INITIALIZATION,'initialization_sha256')]:
            if digest(path)!=report[key]:raise RuntimeError(f'Input changed: {path}')
        for projection,path in report['near_depth_source_paths'].items():
            if digest(path)!=report['near_depth_source_sha256'][projection]:raise RuntimeError('Near-depth source changed')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',
            NCCL_DEBUG='WARN',TORCHINDUCTOR_CACHE_DIR=str(LOCAL/'inductor'),
            TRITON_CACHE_DIR=str(LOCAL/'triton'),TORCH_EXTENSIONS_DIR=str(LOCAL/'torch_extensions'))
        environment.pop('TORCH_COMPILE_DISABLE',None)
        prefix=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4']
        status('GPU_correctness_check_shared_launch',**check_admission())
        code=run_logged(prefix+[str(ROOT/'gpu_smoke.py')],environment,logs/f'gpu_correctness_{stamp}.log')
        if code:
            status('failed_GPU_correctness_check',exit_code=code);return code
        admission=check_admission()
        args=prefix+[str(ROOT/'train.py'),'--wan',report['wan_root'],
            '--initialization-checkpoint',str(INITIALIZATION),'--manifest',str(LOCAL/'train.jsonl'),
            '--offsets',str(LOCAL/'train.offsets.npy'),'--output',str(OUT),'--max-steps',str(FINAL_STEP),
            '--save-every','2000','--ramp-steps',str(RAMP_STEPS),'--nvs-probability','0.5',
            '--path-map',str(ROOT/'image_path_map.shared.json')]
        log=logs/f'train_{stamp}.log'
        (OUT/'launch.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
            command=args,log=str(log),gpu_list='0,1,2,3',shared_admission=admission,
            launcher=str(Path(__file__).resolve()),launcher_sha256=digest(__file__)),indent=2)+'\n')
        status('launching_user_requested_shared_training',log=str(log),**admission)
        code=run_logged(args,environment,log)
        status('complete' if code==0 else 'failed',exit_code=code)
        return code

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        status('immediate_launch_failed',error=repr(exc))
        raise
