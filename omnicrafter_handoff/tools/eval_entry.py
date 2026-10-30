"""Relocate frozen evaluation code without changing its inference/metric math."""
import argparse
import importlib
import sys
from handoff_common import *


def bind(config):
    source = ROOT/'snapshots/evaluation'
    sys.path.insert(0, str(source))
    import eval_shared as shared
    shared.CODE = source
    shared.SNAPSHOT = source/'training_snapshot'
    shared.REPO = shared.SNAPSHOT/'code'
    shared.TOOLS = source
    shared.CHECKPOINT = Path(config['checkpoint'])
    shared.OUT = Path(config['output'])
    shared.WAN = Path(config['wan_root'])
    shared.CACHE = Path(config['cache_dir'])
    shared.DREAM_CACHE = Path(config['dreamsim_cache'])
    shared.FID_PACKAGES = source  # clean-fid is installed in the metric environment.
    shared.FID_SOURCE = Path(config['fid_weights'])
    shared.NUM_SHARDS = config['num_gpus']
    shared.code_hashes = protocol_hashes
    return shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=['worker', 'score', 'aggregate'], required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    bind(read_json(args.config))
    sys.argv = [args.phase + '.py']
    if args.phase != 'aggregate':
        sys.argv += ['--rank', str(args.rank)]
    if args.smoke:
        if args.phase != 'worker':
            raise ValueError('Portable smoke mode is inference-only')
        sys.argv += ['--smoke']
    importlib.import_module(args.phase).main()


if __name__ == '__main__':
    main()
