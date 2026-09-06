"""Compatibility entrypoint for append-safe landing manifests."""
from __future__ import annotations
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "build_landing_manifest.py"), run_name="__main__")
