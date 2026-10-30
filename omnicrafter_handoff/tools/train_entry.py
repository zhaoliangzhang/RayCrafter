"""Bind portable locations before importing the untouched four-GPU trainer."""
import argparse
import sys
from handoff_common import *


def bind(config):
    source = ROOT/'snapshots/mixed'
    sys.path.insert(0, str(source))
    import settings
    settings.OUT = Path(config['output'])
    settings.DATA = Path(config['data_dir'])
    import common
    common.OUT = settings.OUT
    common.DATA = settings.DATA
    common.FINAL_STEP = config['final_step']
    common.SAVE_STEPS = config['save_every']
    common.GPU_UUIDS = []  # Memory parking is intentionally never enabled here.
    common.WAN = Path(config['wan_root'])
    common.PUBLISHED = Path(config['base_initialization'])
    return common


def training_arguments(config):
    architecture = camera_architecture(config)
    base = config.get('prepared_initialization')
    if architecture == 'scope' and not base:
        raise ValueError('Run tools/train_mixed.py to prepare the SCoPE image-only initialization first')
    arguments = ['train.py', '--architecture', architecture, '--wan', config['wan_root'],
                 '--base-initialization', base or config['base_initialization'],
                 '--initial-step', str(config['initial_step'])]
    if config.get('restore_checkpoint'):
        arguments += ['--restore-checkpoint', config['restore_checkpoint']]
    return arguments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    config = read_json(p.parse_args().config)
    arguments = training_arguments(config)
    architecture = camera_architecture(config)
    checkpoint_header(config.get('prepared_initialization') or config['base_initialization'],
                      architecture, initialization=True)
    bind(config)
    import train
    sys.argv = arguments
    train.main()


if __name__ == '__main__':
    main()
