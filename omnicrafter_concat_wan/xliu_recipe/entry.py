"""The handoff's ``tools/train_entry.py`` with collectives that wait 2 hours instead of 10 minutes.

ZeRO's gradient group takes torch's 10-minute NCCL default, and an NFS read stall that long killed the
Wan concat run at step 44134 (961df06). The training code itself is untouched.
"""
import sys
from datetime import timedelta
from pathlib import Path

import torch.distributed.distributed_c10d as c10d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'omnicrafter_handoff/tools'))
import train_entry  # noqa: E402

COLLECTIVE_TIMEOUT = timedelta(hours=2)

if __name__ == '__main__':
    # Groups made without a timeout read this default; the world group's comes from DEEPSPEED_TIMEOUT.
    c10d.default_pg_nccl_timeout = COLLECTIVE_TIMEOUT
    train_entry.main()
