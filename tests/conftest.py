import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLINENV = ROOT / "third_party" / "clinenv"
for p in (str(ROOT), str(CLINENV), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
