"""Compatibility entrypoint for the public fast-path CLI."""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_fast_path.py"), run_name="__main__")
