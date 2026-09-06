"""Compatibility entrypoint for EKS validation."""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "validate_eks.py"), run_name="__main__")
