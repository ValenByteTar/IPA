"""Compatibility entrypoint for web scraping."""
from __future__ import annotations
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_web_scrape.py"), run_name="__main__")
