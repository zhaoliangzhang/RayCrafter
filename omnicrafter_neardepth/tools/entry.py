"""Bind portable locations, then run the untouched snapshot train.py or check_data.py.

    entry.py --config <output>/runtime/train_config.json --script train -- <train.py arguments>
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parents[1]/'snapshot'


def bind(config):
    sys.path.insert(0, str(SNAPSHOT))
    import settings
    settings.OUT = Path(config['output'])
    settings.DATA = Path(config['data_dir'])
    settings.HELDOUT = settings.DATA/'heldout_scenes.json'
    import common
    common.OUT, common.DATA, common.HELDOUT = settings.OUT, settings.DATA, settings.HELDOUT
    common.FINAL_STEP = config['final_step']
    common.SAVE_STEPS = config['save_every']
    common.GPU_UUIDS = []  # Memory parking is never enabled (train.py also refuses it).
    common.WAN = Path(config['wan_root'])
    common.PUBLISHED = Path(config['base_initialization'])
    return common


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--script', choices=['train', 'check_data'], required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    bind(json.loads(args.config.read_text()))
    module = importlib.import_module(args.script)
    sys.argv = [str(SNAPSHOT/f'{args.script}.py'), *[a for a in args.arguments if a != '--']]
    module.main()


if __name__ == '__main__':
    main()
