"""CLI wrapper for Reporter deep dives."""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ipa.reporter.reporter_deep_dive import main

if __name__ == "__main__":
    raise SystemExit(main())
