"""Import paths for the scope-ff run: the unmodified OmniCrafter near-depth snapshot and
its DiffSynth code, the handoff tools, and RayCrafter's ``src``. Import this first."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SNAPSHOT = REPO/'omnicrafter_neardepth/snapshot'
CODE = SNAPSHOT/'code'
HANDOFF_TOOLS = REPO/'omnicrafter_handoff/tools'
SRC = REPO/'src'

# After this file's directory (sys.path[0] for the scripts here), so local modules win.
for path in (SRC, HANDOFF_TOOLS, CODE, SNAPSHOT):
    if str(path) not in sys.path:
        sys.path.insert(1, str(path))
