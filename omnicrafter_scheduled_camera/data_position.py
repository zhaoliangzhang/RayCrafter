"""Each rank's place in the shuffled data, kept in the full state.

A resumed job continues with the same sets and the same task/dropout draws as an
uninterrupted run. The shuffle generator's state at the start of an epoch fixes
that epoch's order. The loader fetches one set ahead, drawing its task from
Python's ``random`` before the current set is trained on, so a mid-epoch state
keeps the RNG from before that fetch (``MixedProjectionDataset`` records it).
"""
import random


def epoch_batches(accelerator, loader, generator, epoch, skip):
    """``loader`` for ``epoch`` from batch ``skip`` on, and the generator state that orders it."""
    start = generator.get_state()
    active = accelerator.skip_first_batches(loader, skip) if skip else loader
    active.set_epoch(epoch)
    return active, start


def data_state(data, generator, epoch, position, batches_per_epoch, epoch_start):
    """The state after ``position`` batches of ``epoch`` have been trained on."""
    if position == batches_per_epoch:
        # Nothing was fetched ahead: the next epoch starts from the current generator.
        return {'epoch': epoch + 1, 'batch_in_epoch': 0, 'generator': generator.get_state(),
                'python_random': random.getstate()}
    return {'epoch': epoch, 'batch_in_epoch': position, 'generator': epoch_start,
            'python_random': data.random_state_before_item}


def restore(generator, state):
    """Apply a ``data_state`` after ``accelerator.load_state``; returns (epoch, skip)."""
    generator.set_state(state['generator'])
    random.setstate(state['python_random'])
    return state['epoch'], state['batch_in_epoch']
