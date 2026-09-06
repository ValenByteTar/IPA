"""Public entrypoint for the Reporter periodic report (thin wrapper over ipa)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from ipa.reporter.reporter_cli import main

if __name__ == "__main__":
    sys.exit(main())
