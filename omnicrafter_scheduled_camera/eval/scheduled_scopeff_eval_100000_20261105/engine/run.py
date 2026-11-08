"""Independent persistent inference/metric shards; never interrupt other jobs."""
import fcntl
import shutil
import signal
import socket
import subprocess
import time
from eval_shared import *

CHILDREN=[]
def state(stage,**values):
    counts={d:{t:len(list((OUT/'bundles'/d/t).glob('*/*/_SUCCESS'))) for t in TASKS} for d in COUNTS}
    value={'time':now(),'model':MODEL,'server':HOST,'stage':stage,'completed_sets':counts,
           'expected_sets_per_task':COUNTS,**values}
    write_json(OUT/'runtime/status.json',value);print(json.dumps(value),flush=True)
def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    rows=[[v.strip() for v in line.split(',')] for line in raw.splitlines()]
    chosen={r[1]:r for r in rows if r[1] in GPUS}
    assert set(chosen)==set(GPUS),'Wrong physical GPUs'
    assert [int(chosen[g][0]) for g in GPUS]==GPU_INDICES
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=20)
    occupied=[v for v in apps.splitlines() if len(v.split(','))>1 and v.split(',')[1].strip() in GPUS]
    return {'idle':not occupied and all(int(r[2])<1024 and int(r[3])<=5 for r in chosen.values()),
            'gpus':list(chosen.values()),'compute_apps':occupied}
def wait_idle(next_stage):
    count=0
    while count<3:
        current=inventory();count=count+1 if current['idle'] else 0
        state('confirming_idle_gpus' if count else 'waiting_for_idle_gpus',next_stage=next_stage,
              consecutive_idle_checks=count,**current)
        if count<3: time.sleep(15 if count else 60)
def spawn(label,script,arguments=(),gpu=None,metric=False):
    env={**os.environ,'OC_EVAL_MODEL':MODEL,'CAMERA_VARIANT':VARIANT,'CUDA_VISIBLE_DEVICES':'' if gpu is None else GPUS[gpu],
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
         'TRITON_CACHE_DIR':str(CACHE/f'triton_{gpu}')}
    if script=='test_eval.py':env['TORCH_COMPILE_DISABLE']='1'
    log=OUT/f'logs/{label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    command=[str(METRIC_PYTHON if metric else PYTHON),'-B','-u',str(CODE/script),*arguments]
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append(child)
    write_json(OUT/f'runtime/command_{label}.json',{'time':now(),'command':command,'log':str(log),'pid':child.pid,
        'gpu_uuid':None if gpu is None else GPUS[gpu]})
    return label,child,log
def wait(jobs,stage):
    while True:
        failed=[(n,p.returncode,str(log)) for n,p,log in jobs if p.poll() not in (None,0)]
        if failed: raise RuntimeError(f'{stage} failed: {failed}')
        running=[n for n,p,_ in jobs if p.poll() is None];state(stage,running=running)
        if not running: return
        time.sleep(30)
def cache():
    CACHE.mkdir(parents=True,exist_ok=True);(CACHE/'fid').mkdir(exist_ok=True)
    for source,dest in [(CHECKPOINT,CACHE/CHECKPOINT.name),(FID_SOURCE,CACHE/'fid/inception-2015-12-05.pt')]:
        if not dest.exists():
            temp=dest.with_name(f'.{dest.name}.{os.getpid()}.tmp');shutil.copy2(source,temp);temp.replace(dest)
        assert sha256(source)==sha256(dest)
    if not DREAM_CACHE.exists():
        if HOST=='io96': link(DREAM_SOURCE,DREAM_CACHE)
        else: shutil.copytree(DREAM_SOURCE,DREAM_CACHE)
    model=WAN/'Wan-AI/Wan2.1-I2V-14B-480P'
    weights=sorted(model.glob('diffusion_pytorch_model-*-of-00007.safetensors'))
    assert len(weights)==7
    weights += [model/name for name in ('models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    canonical=json.loads((MATCHED/'runtime/local_cache.json').read_text())
    assert {p.name:p.stat().st_size for p in weights}=={Path(p).name:s for p,s in canonical['wan_files'].items()}
    assert sha256(CACHE/'fid/inception-2015-12-05.pt')==canonical['fid_sha256']
    assert (WAN/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json').is_file()
    expected=json.loads((CODE/'expected_wan_sha256.json').read_text());verified={}
    assert len(expected)==len(weights)==10
    for path in weights:
        digest=sha256(path);assert digest==expected[path.name],f'WAN hash mismatch: {path}'
        verified[path.name]=digest
        print(json.dumps({'time':now(),'verified_wan':path.name}),flush=True)
    write_json(OUT/'runtime/local_cache.json',{'time':now(),'checkpoint_sha256':sha256(CACHE/CHECKPOINT.name),
        'fid_sha256':canonical['fid_sha256'],'wan_files':{str(p):p.stat().st_size for p in weights},
        'dreamsim_cache':str(DREAM_CACHE),'wan_sha256':verified})
def main():
    assert MODEL in ('scope','rayrope') and socket.gethostname().split('.')[0]==HOST
    for name in ('runtime','logs','metrics'): (OUT/name).mkdir(parents=True,exist_ok=True)
    with (OUT/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inventory()
        while not (BASE/'benchmark/_SUCCESS').is_file():
            state('waiting_for_shared_preparation');time.sleep(30)
        assert code_hashes()==json.loads((OUT/'provenance.json').read_text())['code_sha256']
        wait([spawn('cpu_eval_tests','test_eval.py')],'cpu_tests')
        state('preparing_local_cache');cache()
        wait_idle('inference_smoke')
        wait([spawn('inference_smoke','worker.py',['--rank','0','--smoke'],gpu=0)],'inference_smoke')
        wait_idle('metric_smoke')
        wait([spawn('metric_smoke','score.py',['--rank','0','--smoke'],gpu=NUM_SHARDS-1,metric=True)],'metric_smoke')
        wait_idle('inference')
        wait([spawn(f'inference_rank{r}','worker.py',['--rank',str(r)],gpu=r) for r in range(NUM_SHARDS)],'inference')
        wait_idle('metrics')
        wait([spawn(f'metrics_rank{r}','score.py',['--rank',str(r)],gpu=r,metric=True) for r in range(NUM_SHARDS)],'metrics')
        wait([spawn('aggregate','aggregate.py',metric=True)],'aggregating')
        assert (OUT/'_SUCCESS').is_file()
        wait([spawn('comparison','comparison.py',metric=True)],'reporting')
        state('complete',summary=str(OUT/'metrics/summary.json'))
def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try: main()
    except BlockingIOError: raise SystemExit('Existing controller owns evaluation')
    except BaseException as exc:
        for child in CHILDREN:
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        state('failed',error=repr(exc));raise
