"""Import paths for the UniRoPE run: scope-ff's (the unmodified OmniCrafter near-depth snapshot,
the handoff tools, RayCrafter's ``src``), plus ``omnicrafter_scopeff`` itself. Import this first."""
import sys
from pathlib import Path

SCOPEFF = Path(__file__).resolve().parent.parent/'omnicrafter_scopeff'
# After this file's directory (sys.path[0] for the scripts here), so local modules win.
if str(SCOPEFF) not in sys.path:
    sys.path.insert(1, str(SCOPEFF))

import scopeff_paths  # noqa: E402,F401
