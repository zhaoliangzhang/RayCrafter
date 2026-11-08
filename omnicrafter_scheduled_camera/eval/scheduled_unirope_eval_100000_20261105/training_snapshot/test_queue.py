import unittest
from config import GPU_UUIDS
from queue_training import parse_probe

class QueueTests(unittest.TestCase):
    def inventory(self,used=0,util=0):
        return '\n'.join(f'{i}, {uuid}, {used}, {util}' for i,uuid in enumerate(GPU_UUIDS))
    def test_idle(self):
        self.assertTrue(parse_probe(self.inventory(),'')['idle'])
    def test_process_even_small_blocks_launch(self):
        self.assertFalse(parse_probe(self.inventory(),f'{GPU_UUIDS[0]}, 123')['idle'])
    def test_memory_holder_blocks_launch(self):
        self.assertFalse(parse_probe(self.inventory(60000,0),'')['idle'])
    def test_utilization_blocks_launch(self):
        self.assertFalse(parse_probe(self.inventory(0,80),'')['idle'])
    def test_wrong_host_rejected(self):
        with self.assertRaises(RuntimeError):parse_probe(self.inventory().replace(GPU_UUIDS[0],'GPU-WRONG'),'')

if __name__=='__main__':unittest.main(verbosity=2)
