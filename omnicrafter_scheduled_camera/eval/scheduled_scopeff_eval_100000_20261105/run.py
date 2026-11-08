"""Persistent io107 evaluation controller with smoke checks and resumable shards."""
import importlib.util
import signal
import socket
from eval_shared import *
spec=importlib.util.spec_from_file_location('evaluation_engine',ENGINE/'run.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def main():
    assert socket.gethostname().split('.')[0]==HOST
    for name in ('runtime','logs','metrics'):(OUT/name).mkdir(parents=True,exist_ok=True)
    with (OUT/'runtime/controller.lock').open('a') as lock:
        engine.fcntl.flock(lock,engine.fcntl.LOCK_EX|engine.fcntl.LOCK_NB)
        engine.inventory()
        if (OUT/'_SUCCESS').is_file():
            engine.state('complete',summary=str(OUT/'metrics/summary.json'));return
        if not (OUT/'benchmark/_SUCCESS').is_file():
            engine.wait([engine.spawn('prepare','prepare.py')],'preparing')
        assert code_hashes()==json.loads((OUT/'provenance.json').read_text())['code_sha256']
        engine.wait([engine.spawn('cpu_tests','test_eval.py')],'cpu_tests')
        engine.state('preparing_local_cache');engine.cache()
        engine.wait_idle('inference_smoke')
        engine.wait([engine.spawn('inference_smoke','worker.py',['--rank','0','--smoke'],gpu=0)],'inference_smoke')
        engine.wait_idle('metric_smoke')
        engine.wait([engine.spawn('metric_smoke','score.py',['--rank','0','--smoke'],gpu=NUM_SHARDS-1,metric=True)],'metric_smoke')
        engine.wait_idle('inference')
        engine.wait([engine.spawn(f'inference_rank{r}','worker.py',['--rank',str(r)],gpu=r) for r in range(NUM_SHARDS)],'inference')
        engine.wait_idle('metrics')
        engine.wait([engine.spawn(f'metrics_rank{r}','score.py',['--rank',str(r)],gpu=r,metric=True) for r in range(NUM_SHARDS)],'metrics')
        engine.wait([engine.spawn('aggregate','aggregate.py',metric=True)],'aggregating')
        assert (OUT/'_SUCCESS').is_file()
        engine.state('complete',summary=str(OUT/'metrics/summary.json'))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,engine.stop);signal.signal(signal.SIGINT,engine.stop)
    try:main()
    except BlockingIOError:raise SystemExit('Another controller owns this evaluation')
    except BaseException as exc:
        for p in engine.CHILDREN:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        engine.state('failed',error=repr(exc));raise
