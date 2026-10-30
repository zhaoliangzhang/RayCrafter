"""Stage WAN in this experiment's persistent io100 disk cache."""
import shutil
import subprocess
import fcntl
from common import *


def stage():
    if (WAN/'_READY.json').is_file():
        model_paths(WAN)
        print('Local WAN cache ready',flush=True)
        return
    source_model = WAN_FALLBACK/'Wan-AI/Wan2.1-I2V-14B-480P'
    source_tokenizer = WAN_FALLBACK/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    model_paths(WAN_FALLBACK)
    size = sum(p.stat().st_size for p in source_model.rglob('*') if p.is_file())
    WAN.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(WAN).free > size + 32*1024**3
    # Optional read-only reuse of our already verified cache on io104. This
    # avoids the slower shared-home route; all file sizes are still checked
    # against the canonical checkpoint directory below.
    remote_root = os.environ.get('OC_WAN_STAGE_REMOTE')
    if remote_root:
        assert remote_root == 'io104:/dev/shm/xliu253/framecrafter_wan21'
    for relative, source in [('Wan-AI/Wan2.1-I2V-14B-480P',source_model),
                             ('Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl',source_tokenizer)]:
        destination = WAN/relative
        destination.mkdir(parents=True,exist_ok=True)
        transfer_source = f'{remote_root}/{relative}' if remote_root else str(source)
        print(f'{now()} Staging {transfer_source} -> {destination}',flush=True)
        subprocess.run(['rsync','-a','--partial','--info=progress2',
                        '-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
                        transfer_source+'/',str(destination)+'/'],check=True)
        for path in source.rglob('*'):
            if path.is_file():
                assert (destination/path.relative_to(source)).stat().st_size == path.stat().st_size
    model_paths(WAN)
    write_json(WAN/'_READY.json',{'time':now(),'source':str(WAN_FALLBACK),'cache':str(WAN),
        'transfer_source':remote_root or str(WAN_FALLBACK),
        'transfer':'rsync checksummed transfer; complete model and tokenizer; file sizes verified'})
    print('Local WAN staging complete',flush=True)


def main():
    WAN.parent.mkdir(parents=True,exist_ok=True)
    with (WAN.parent/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stage()


if __name__ == '__main__':
    main()
