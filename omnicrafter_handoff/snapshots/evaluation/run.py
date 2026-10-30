"""Persistent seven-GPU io95 evaluation; GPU 7 is never used or interrupted."""
import fcntl
import shutil
import signal
import subprocess
import time
from eval_shared import *

CHILDREN=[]

def state(stage,**values):
    counts={d:{t:len(list((OUT/'bundles'/d/t).glob('*/*/_SUCCESS'))) for t in TASKS} for d in COUNTS}
    record={'time':now(),'stage':stage,'completed_sets':counts,'expected_sets_per_task':COUNTS,**values}
    write_json(OUT/'runtime/status.json',record);print(json.dumps(record),flush=True)

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    rows=[[v.strip() for v in line.split(',')] for line in raw.splitlines()]
    selected={r[0]:r for r in rows if r[0] in GPUS}
    assert set(selected)==set(GPUS),'Incorrect host or selected GPU identities'
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=20).strip()
    occupied=[line for line in apps.splitlines() if line.split(',')[1].strip() in GPUS]
    idle=not occupied and all(int(r[1])<1024 and int(r[2])<=5 for r in selected.values())
    return {'idle':idle,'selected_gpus':[selected[uuid] for uuid in GPUS],'selected_compute_apps':occupied}

def gpu_check():
    assert gpu_inventory()['idle'],'Selected GPUs occupied; no existing job will be stopped'

def wait_for_gpus(next_stage):
    consecutive=0
    while consecutive<3:
        current=gpu_inventory()
        consecutive=consecutive+1 if current['idle'] else 0
        interval=15 if consecutive else 60
        state('confirming_idle_gpus' if consecutive else 'waiting_for_seven_idle_gpus',
              next_stage=next_stage,poll_seconds=interval,queue_timeout=None,
              consecutive_idle_checks=consecutive,**current)
        if consecutive<3:
            time.sleep(interval)

def spawn(label,script,arguments=(),gpu=None,metric=False):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'' if gpu is None else GPUS[gpu],
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
         'TRITON_CACHE_DIR':f'/export/io95/data/xliu253/triton_cache/mixed50000_eval_{gpu}'}
    path=OUT/f'logs/{label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    command=[str(METRIC_PYTHON if metric else PYTHON),'-B','-u',str(CODE/script),*arguments]
    with path.open('x') as handle:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append(child)
    write_json(OUT/f'runtime/command_{label}.json',{'time':now(),'command':command,'log':str(path),'gpu_uuid':None if gpu is None else GPUS[gpu],'pid':child.pid})
    return label,child,path

def wait(jobs,stage):
    while True:
        failed=[(label,p.returncode,str(path)) for label,p,path in jobs if p.poll() not in (None,0)]
        if failed: raise RuntimeError(f'{stage} failed: {failed}')
        running=[label for label,p,path in jobs if p.poll() is None]
        state(stage,running=running)
        if not running: return
        time.sleep(30)

def cache():
    CACHE.mkdir(parents=True,exist_ok=True);(CACHE/'fid').mkdir(exist_ok=True)
    for source,dest in [(CHECKPOINT,CACHE/CHECKPOINT.name),(FID_SOURCE,CACHE/'fid/inception-2015-12-05.pt')]:
        if not dest.exists():
            temp=dest.with_name(f'.{dest.name}.{os.getpid()}.tmp');shutil.copy2(source,temp);temp.replace(dest)
        assert sha256(source)==sha256(dest),(source,dest)
    model=WAN/'Wan-AI/Wan2.1-I2V-14B-480P'
    shards=sorted(model.glob('diffusion_pytorch_model-*-of-00007.safetensors'))
    assert len(shards)==7
    weights=shards+[model/name for name in ('models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    assert all(p.is_file() and p.stat().st_size>1000000 for p in weights)
    previous_cache=json.loads((MATCHED/'runtime/local_cache.json').read_text())
    assert {p.name:p.stat().st_size for p in weights}=={Path(p).name:size for p,size in previous_cache['wan_files'].items()}
    assert sha256(CACHE/'fid/inception-2015-12-05.pt')==previous_cache['fid_sha256']
    assert (WAN/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json').is_file()
    assert DREAM_CACHE.is_dir() and (FID_PACKAGES/'cleanfid').is_dir() and METRIC_PYTHON.is_file()
    write_json(OUT/'runtime/local_cache.json',{'time':now(),'checkpoint_sha256':sha256(CACHE/CHECKPOINT.name),
        'fid_sha256':sha256(CACHE/'fid/inception-2015-12-05.pt'),
        'wan_files':{str(p):p.stat().st_size for p in weights},'dreamsim_cache':str(DREAM_CACHE)})

def main():
    for name in ('runtime','logs','metrics'): (OUT/name).mkdir(parents=True,exist_ok=True)
    with (OUT/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_inventory()  # Check host identities; CPU preparation does not require idle GPUs.
        wait([spawn('cpu_protocol_tests','training_snapshot/test_protocol.py'),spawn('cpu_eval_tests','test_eval.py')],'cpu_tests')
        if (OUT/'benchmark/_SUCCESS').is_file():
            assert code_hashes()==json.loads((OUT/'provenance.json').read_text())['code_sha256']
        else:
            wait([spawn('prepare','prepare_matched.py')],'preparing_benchmark')
        state('preparing_local_cache');cache();wait_for_gpus('inference_smoke')
        wait([spawn('inference_smoke','worker.py',['--rank','0','--smoke'],gpu=0)],'inference_smoke')
        wait_for_gpus('metric_smoke')
        wait([spawn('metric_smoke','score.py',['--rank','0','--smoke'],gpu=NUM_SHARDS-1,metric=True)],'metric_smoke')
        wait_for_gpus('inference')
        wait([spawn(f'inference_rank{rank}','worker.py',['--rank',str(rank)],gpu=rank) for rank in range(NUM_SHARDS)],'inference')
        wait_for_gpus('metrics')
        wait([spawn(f'metrics_rank{rank}','score.py',['--rank',str(rank)],gpu=rank,metric=True) for rank in range(NUM_SHARDS)],'metrics')
        wait([spawn('aggregate','aggregate.py',metric=True)],'aggregating_fid_and_metrics')
        assert (OUT/'_SUCCESS').is_file()
        state('complete',summary=str(OUT/'metrics/summary.json'))

def stop(signum,frame):
    raise KeyboardInterrupt(f'Signal {signum}')

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        main()
    except BlockingIOError:
        raise SystemExit('Another controller owns this evaluation')
    except BaseException as exc:
        for child in CHILDREN:
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        state('failed',error=repr(exc));raise
