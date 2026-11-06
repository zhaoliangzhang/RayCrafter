"""A resume mid-epoch or at an epoch end continues the same sets and draws (CPU, no WAN)."""
import random
import unittest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from data_position import epoch_batches, data_state, restore


class Draws(Dataset):
    """Draws from Python's ``random`` per set, as the task/dropout draw does."""
    def __len__(self):
        return 7

    def __getitem__(self, index):
        self.random_state_before_item = random.getstate()
        return index, random.random()


class DataPositionTests(unittest.TestCase):
    def setUp(self):
        self.accelerator = Accelerator(cpu=True)

    def sets(self, total, python_seed, state=None, stop_after=None):
        """The loop of ``train.py``: the sets seen, and the data state after ``stop_after`` of them."""
        data = Draws()
        generator = torch.Generator().manual_seed(20261023)
        loader = self.accelerator.prepare(DataLoader(data, batch_size=1, shuffle=True, drop_last=True,
                                                     collate_fn=lambda rows: rows, generator=generator))
        # A new process: Python's RNG differs until the state is restored.
        random.seed(python_seed)
        epoch, skip = restore(generator, state) if state else (0, 0)
        seen = []
        while len(seen) < total:
            active, epoch_start = epoch_batches(self.accelerator, loader, generator, epoch, skip)
            position = skip
            for batch in active:
                seen.append(batch[0])
                position += 1
                if len(seen) == stop_after:
                    return seen, data_state(data, generator, epoch, position, len(loader), epoch_start)
                if len(seen) == total:
                    return seen, None
            epoch, skip = epoch + 1, 0
        return seen, None

    def test_resume_matches_uninterrupted(self):
        full, _ = self.sets(21, python_seed=1)
        self.assertEqual(sorted(i for i, _ in full[7:14]), list(range(7)))
        self.assertNotEqual([i for i, _ in full[:7]], [i for i, _ in full[7:14]])
        # Mid-epoch 0, end of epoch 0, mid-epoch 1, end of epoch 1, mid-epoch 2.
        for stop_after in (3, 7, 9, 14, 20):
            first, state = self.sets(21, python_seed=1, stop_after=stop_after)
            rest, _ = self.sets(21 - stop_after, python_seed=999, state=state)
            self.assertEqual(first + rest, full, f'resume after {stop_after} sets')


if __name__ == '__main__':
    unittest.main(verbosity=2)
