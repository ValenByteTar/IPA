"""Public entrypoint for the EXP-003 agentic vs legacy evaluation (thin wrapper over ipa)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from ipa.agentic.agentic_eval import main

if __name__ == "__main__":
    sys.exit(main())
