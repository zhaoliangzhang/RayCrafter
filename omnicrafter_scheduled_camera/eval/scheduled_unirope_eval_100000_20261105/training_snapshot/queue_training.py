"""Non-preemptive four-card queue. Poll every 60 seconds with no expiry.

Requires three consecutive idle probes AND an immediate pre-launch recheck.
Does not kill processes, allocate memory holders, or override occupied GPUs.
"""
import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from config import ROOT,HOST,OUT,PYTHON,GPU_UUIDS,VARIANT

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()

def command(args):
    return subprocess.run(args,check=True,text=True,capture_output=True,timeout=30).stdout

def parse_probe(gpu_text,process_text):
    rows=list(csv.reader(gpu_text.strip().splitlines(),skipinitialspace=True))
    if len(rows)!=4 or [r[1] for r in rows]!=GPU_UUIDS or [int(r[0]) for r in rows]!=list(range(4)):
        raise RuntimeError(f'GPU identities do not match authorized {HOST} cards')
    cards=[dict(index=int(r[0]),uuid=r[1],used_mib=int(r[2]),utilization=int(r[3])) for r in rows]
    processes=[]
    for r in csv.reader(process_text.strip().splitlines(),skipinitialspace=True):
        if not r:
            continue
        if len(r)!=2 or r[0] not in GPU_UUIDS:
            raise RuntimeError(f'Unexpected process row: {r}')
        processes.append(dict(uuid=r[0],pid=int(r[1])))
    return dict(idle=not processes and all(c['used_mib']<=1024 and c['utilization']<=5 for c in cards),
                gpus=cards,compute_processes=processes)

def probe():
    return parse_probe(command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
        command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']))

def status(stage,**extra):
    record=dict(time=datetime.now().astimezone().isoformat(),stage=stage,pid=os.getpid(),host=HOST,
                variant=VARIANT,gpu_list='0,1,2,3',poll_seconds=60,wait_timeout=None,**extra)
    temporary=OUT/f'.queue_status.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(OUT/'queue_status.json')
    print(json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        result=probe();print(json.dumps(result),flush=True)
        return 0 if result['idle'] else 75
    OUT.mkdir(parents=True,exist_ok=True)
    if list(OUT.glob('step-*.safetensors')):
        raise RuntimeError('Existing run found; refusing an implicit restart/overwrite')
    with (OUT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        logs=OUT/'logs';logs.mkdir(exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        queue_log=(logs/f'queue_{stamp}.log').open('a',buffering=1)
        sys.stdout=Tee(sys.stdout,queue_log)
        probe()  # Validate host/cards before any work.
        status('CPU_preflight_no_GPU_allocation')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',TORCH_COMPILE_DISABLE='1')
        with (logs/f'preflight_{stamp}.log').open('a',buffering=1) as preflight_log:
            child=subprocess.Popen([PYTHON,'-B','-u','preflight.py'],cwd=ROOT,env=environment,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                preflight_log.write(line);print(line,end='',flush=True)
            result=subprocess.CompletedProcess(child.args,child.wait())
        if result.returncode:
            status('preflight_failed',exit_code=result.returncode);return result.returncode
        stable=0
        while True:
            try:
                result=probe()
            except Exception as exc:
                stable=0;status('waiting_probe_error',error=repr(exc))
            else:
                stable=stable+1 if result['idle'] else 0
                status('waiting_for_four_idle_gpus',consecutive_idle_checks=stable,**result)
                if stable>=3:
                    with Path(f'/tmp/xliu253_{HOST}_four_gpu_training.lock').open('a') as host_lock:
                        try:
                            fcntl.flock(host_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            stable=0;status('waiting_for_launcher_lock')
                        else:
                            if not probe()['idle']:
                                stable=0;continue
                            status('launching')
                            completed=subprocess.run([PYTHON,'-B','-u','run_job.py'],cwd=ROOT)
                            if completed.returncode==75:
                                stable=0;status('waiting_GPU_became_busy_before_launch')
                            else:
                                status('complete' if completed.returncode==0 else 'failed',exit_code=completed.returncode)
                                return completed.returncode
            time.sleep(60)

if __name__=='__main__':
    sys.exit(main())
