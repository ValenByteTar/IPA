"""Compatibility entrypoint for manifest validation."""
from __future__ import annotations
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "validate_contracts.py"), run_name="__main__")
