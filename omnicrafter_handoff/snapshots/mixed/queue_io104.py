"""Non-preempting queue: require configuration, validated data, four idle io104 GPUs."""
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from common import *

RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
CHILD = None
MONITOR = None
RESUME_STEP = int(os.environ['OC_REQUIRE_RESUME_STEP']) if os.environ.get('OC_REQUIRE_RESUME_STEP') else None
HEADROOM = int(os.environ.get('OC_RESERVE_HEADROOM_GIB','0'))
LOCAL = Path('/export/io104/data/xliu253/mixed_projection_runtime') / RUN_ID


def validate_resume():
    if RESUME_STEP is None:
        return None
    from safetensors import safe_open
    checkpoints = sorted(OUT.glob('step-*.safetensors'),key=lambda p:int(p.stem.split('-')[1]))
    source = OUT/f'step-{RESUME_STEP}.safetensors'
    assert checkpoints and checkpoints[-1] == source,'Requested resume must be the latest checkpoint; refusing overwrite/restart ambiguity'
    with safe_open(str(source),framework='pt',device='cpu') as checkpoint:
        assert checkpoint.metadata()['step'] == str(RESUME_STEP)
        assert len(checkpoint.keys()) == 802
        assert checkpoint.get_slice('patch_embedding.weight').get_shape() == [5120,420,1,2,2]
    return sha256(source)


def state(stage,**values):
    record = {'time':now(),'stage':stage,'server':'io104','run_id':RUN_ID,**values}
    write_json(OUT/'runtime/queue_status.json',record)
    print(json.dumps(record,sort_keys=True),flush=True)


def inventory():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    rows = [[x.strip() for x in line.split(',')] for line in raw.strip().splitlines()]
    assert len(rows)==4 and [r[1] for r in rows]==GPU_UUIDS,'Unexpected GPU identities'
    applications = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=20).strip()
    return {'idle':not applications and all(int(r[3])<1024 and int(r[4])<=5 for r in rows),
        'gpus':rows,'compute_apps':applications}


def wait_for_inputs():
    while True:
        request_path = OUT/'request.json'
        if not request_path.is_file():
            state('waiting_for_configuration',missing=['camera injection: scope or concat','published initialization or checkpoint path'],
                data_ready=(DATA/'_SUCCESS').exists(),poll_seconds=60)
        elif not (DATA/'_SUCCESS').is_file():
            preparation = json.loads((DATA/'preparation_status.json').read_text()) if (DATA/'preparation_status.json').exists() else None
            state('waiting_for_validated_manifest',preparation=preparation,poll_seconds=60)
        else:
            return json.loads(request_path.read_text())
        time.sleep(60)


def prepare(request):
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    architecture = request['camera_injection']
    assert architecture in ('scope','concat')
    assert request['nvs_probability']==.5 and request['steps_in_this_run']==FINAL_STEP
    assert request['batch_size_per_gpu']==1 and request['save_steps']==SAVE_STEPS
    source = Path(request['initialization_path'])
    assert sha256(source)==request['initialization_sha256']
    assert sha256(PUBLISHED)==PUBLISHED_SHA
    summary = json.loads((DATA/'summary.json').read_text())
    assert summary['dataset_counts']==COUNTS and summary['logical_sets']==200000
    assert summary['projection_counts']=={'panorama':100000,'perspective':100000}
    assert sha256(DATA/'train.jsonl')==summary['sha256']
    assert sha256(DATA/'train.offsets.npy')==summary['offsets_sha256']
    if shutil.disk_usage(OUT).free < 160*1024**3:
        raise RuntimeError('Need at least 160 GiB free for 150 checkpoints')
    import yaml
    config = yaml.safe_load((CODE/'zero2.yaml').read_text())
    assert config['deepspeed_config']['zero_stage']==2
    assert config['deepspeed_config']['offload_optimizer_device']==config['deepspeed_config']['offload_param_device']=='none'
    assert config['num_processes']==4
    base = PUBLISHED
    if architecture=='scope':
        base = OUT/'initialization/framecrafter_image36.safetensors'
        state_dict = load_file(str(PUBLISHED),device='cpu')
        assert len(state_dict)==802 and state_dict['patch_embedding.weight'].shape[1]==420
        state_dict['patch_embedding.weight'] = state_dict['patch_embedding.weight'][:,:36].contiguous()
        if base.exists():
            existing = load_file(str(base),device='cpu')
            assert set(existing)==set(state_dict) and all(existing[k].equal(v) for k,v in state_dict.items())
        else:
            temporary = base.with_name(f'.{base.name}.{os.getpid()}.tmp')
            save_file(state_dict,str(temporary),metadata={'source_sha256':PUBLISHED_SHA,'conversion':'preserve LoRA and image channels 0:36; SCoPE supplies camera conditioning'})
            temporary.replace(base)
    restore = None if request['initialization_kind']=='published' else source
    if restore:
        with safe_open(str(restore),framework='pt',device='cpu') as checkpoint:
            assert len(checkpoint.keys())==(1202 if architecture=='scope' else 802)
            assert checkpoint.get_slice('patch_embedding.weight').get_shape()[1]==(36 if architecture=='scope' else 420)
    protocol = {**request,'output_directory':str(OUT),'manifest':str(DATA/'train.jsonl'),
        'manifest_sha256':summary['sha256'],'dataset_counts':COUNTS,'resolutions_width_height':summary['resolutions_width_height'],
        'reference_count':2,'query_count':3,'global_batch_size':4,'gradient_accumulation_steps':1,
        'zero_stage':2,'optimizer':'torch AdamW','optimizer_device':'GPU','cpu_optimizer_offload':False,
        'cpu_activation_offload':False,'initialization_device':'CPU','precision':'bf16',
        'learning_rate':1e-5,'weight_decay':.01,'dropout_percent_in_cleanup':[0,100],
        'pose_gauge':'last query at origin; mean camera-center distance scale',
        'perspective_rays':'scaled K; inverse supplied OpenCV radial/tangential distortion',
        'panorama_rays':'unchanged ERP Pluecker ray formula',
        'optimizer_steps':300000,'step_numbering':'new experiment 0 to 300000',
        'resume_semantics':'adapter/patch/camera weights; optimizer and scheduler restart',
        'base_initialization':str(base),'base_initialization_sha256':sha256(base),
        'queue_poll_seconds':60,'queue_timeout':None,'preemption':False,'oom_fallback':False}
    destination = OUT/'run_config.json'
    if destination.exists():
        assert json.loads(destination.read_text())==protocol,'Existing training protocol differs'
    else:
        write_json(destination,protocol)
    hashes = code_hashes()
    for relative,digest in hashes.items():
        destination = OUT/'provenance'/RUN_ID/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CODE/relative,destination)
        assert sha256(destination)==digest
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(REPO)}
    for name,arguments in [('test_runtime_guard.py',[]),('test_protocol.py',[]),('check_data.py',['--architecture',architecture])]:
        state('preflight',check=name)
        with (OUT/f'logs/{Path(name).stem}_{RUN_ID}.log').open('x') as handle:
            subprocess.run([str(PYTHON),'-B','-u',str(CODE/name),*arguments],cwd=CODE,
                env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    write_json(OUT/f'runtime/preflight_{RUN_ID}.json',{'time':now(),'passed':True,'code_sha256':hashes})
    return protocol,hashes,base,restore


def main():
    global CHILD, MONITOR
    assert HEADROOM == 0 or HEADROOM >= 16
    for name in ('logs','runtime','reports','provenance','initialization'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    with (OUT/'runtime/queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory()
        request = wait_for_inputs()
        protocol,hashes,base,restore = prepare(request)
        resume_sha = validate_resume()
        LOCAL.mkdir(parents=True,exist_ok=True)
        shared_diagnostics = OUT/'runtime/diagnostics'/RUN_ID
        write_json(OUT/f'runtime/relaunch_{RUN_ID}.json',{
            'time':now(),'requested_resume_step':RESUME_STEP,'checkpoint_sha256':resume_sha,
            'final_step':FINAL_STEP,'save_steps':SAVE_STEPS,'headroom_gib':HEADROOM,
            'reserve_after_steps':100,'reservation':'in training process; empty uint8 chunks; no recurrent compute',
            'low_headroom_release_gib':HEADROOM/2,'exclusive_gpu_lock':False,
            'local_diagnostics':str(LOCAL),'shared_diagnostics':str(shared_diagnostics),
            'gpu_monitor_seconds':10,'resume_semantics':'weights/step restored; optimizer/scheduler/data sequence restart'})
        with (LOCAL/'monitor.log').open('x') as monitor_log:
            MONITOR = subprocess.Popen([str(PYTHON),'-B','-u',str(CODE/'monitor_runtime.py'),
                '--local-root',str(LOCAL),'--shared-root',str(shared_diagnostics),
                '--parent-pid',str(os.getpid())],stdout=monitor_log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        while True:
            consecutive = 0
            while consecutive<3:
                try:
                    current = inventory()
                except subprocess.SubprocessError as exc:
                    state('gpu_query_retry',error=repr(exc),poll_seconds=60)
                    consecutive = 0
                    time.sleep(60)
                    continue
                consecutive = consecutive+1 if current['idle'] else 0
                state('confirming_idle_gpus' if consecutive else 'waiting_for_four_idle_gpus',
                    consecutive_idle_checks=consecutive,poll_seconds=15 if consecutive else 60,**current)
                if consecutive<3:
                    time.sleep(15 if consecutive else 60)
            assert code_hashes()==hashes,'Code changed while queued; refusing to launch'
            assert json.loads((OUT/'request.json').read_text())==request
            assert sha256(DATA/'train.jsonl')==protocol['manifest_sha256']
            assert sha256(base)==protocol['base_initialization_sha256']
            assert validate_resume()==resume_sha,'Resume checkpoint changed while queued'
            if restore:
                assert sha256(restore)==request['initialization_sha256']
            if inventory()['idle']:
                break
            state('allocation_changed_retrying',poll_seconds=60)
            time.sleep(60)
        checkpoints = sorted(OUT.glob('step-*.safetensors'),key=lambda p:int(p.stem.split('-')[1]))
        initial_step = 0
        if checkpoints:
            restore = checkpoints[-1]
            initial_step = int(restore.stem.split('-')[1])
            if initial_step==FINAL_STEP:
                assert json.loads((OUT/'_SUCCESS').read_text())['step']==FINAL_STEP
                state('complete',step=FINAL_STEP)
                return
        if RESUME_STEP is not None:
            assert initial_step == RESUME_STEP
        wan = WAN if WAN.is_dir() else WAN_FALLBACK
        model_paths(wan)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        command = [str(PYTHON),'-m','accelerate.commands.launch','--config_file',str(CODE/'zero2.yaml'),
            '--num_processes','4','--main_process_port',str(port),str(CODE/'train.py'),
            '--architecture',request['camera_injection'],'--wan',str(wan),'--base-initialization',str(base),
            '--initial-step',str(initial_step)]
        if restore:
            command += ['--restore-checkpoint',str(restore)]
        if HEADROOM:
            command += ['--reserve-gpu-headroom-gib',str(HEADROOM)]
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':','.join(GPU_UUIDS),'OMP_NUM_THREADS':'4',
            'MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONPATH':str(REPO),
            'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled',
            'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
            'TRITON_CACHE_DIR':'/export/io104/data/xliu253/triton_cache/mixed_projection_2ref3',
            'OC_LOCAL_RUNTIME':str(LOCAL),
            'TORCH_NCCL_TRACE_BUFFER_SIZE':'4096','TORCH_NCCL_DUMP_ON_TIMEOUT':'1',
            'TORCH_NCCL_DEBUG_INFO_TEMP_FILE':str(LOCAL/'nccl_trace_rank_'),
            'TORCH_FR_DUMP_TEMP_FILE':str(LOCAL/'nccl_trace_rank_'),
            'NCCL_DEBUG':'WARN','NCCL_DEBUG_FILE':str(LOCAL/'nccl_%h_%p.log')}
        log = OUT/f'logs/train_{RUN_ID}.log'
        write_json(OUT/f'runtime/command_{RUN_ID}.json',{'command':command,'gpu_uuids':GPU_UUIDS})
        state('launching',log=str(log),initial_step=initial_step,target_step=FINAL_STEP,
            headroom_gib=HEADROOM,local_diagnostics=str(LOCAL))
        with log.open('x') as handle:
            CHILD = subprocess.Popen(command,cwd=CODE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            result = CHILD.wait()
        assert result==0,f'Training exited {result}; inspect {log}; recipe not changed automatically'
        assert (OUT/f'step-{FINAL_STEP}.safetensors').is_file()
        assert json.loads((OUT/'_SUCCESS').read_text())['step']==FINAL_STEP
        state('complete',step=FINAL_STEP)


def interrupted(signum,frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
    raise KeyboardInterrupt(f'Queue received signal {signum}')


def stop_monitor():
    if MONITOR is not None and MONITOR.poll() is None:
        MONITOR.terminate()
        try:
            MONITOR.wait(timeout=15)
        except subprocess.TimeoutExpired:
            MONITOR.kill()
            MONITOR.wait()
    # Only our run-specific diagnostics are copied; never delete local evidence.
    if LOCAL.is_dir():
        try:
            shutil.copytree(LOCAL,OUT/'runtime/diagnostics'/RUN_ID,dirs_exist_ok=True)
        except OSError as exc:
            print(f'Diagnostics remain at {LOCAL}: {exc}',flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGINT,interrupted)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        main()
    except BlockingIOError:
        raise SystemExit('Another controller owns this experiment; not launching a duplicate')
    except BaseException as exc:
        state('failed',error=repr(exc))
        raise
    finally:
        stop_monitor()
