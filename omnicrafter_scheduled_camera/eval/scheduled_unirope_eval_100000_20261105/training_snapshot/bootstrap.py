"""Pinned concat-only WAN snapshot; never import a live training repository."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
