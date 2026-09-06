"""Compatibility entrypoint for adaptive retrieval comparison."""
from __future__ import annotations
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "compare_adaptive_retrieval.py"), run_name="__main__")
