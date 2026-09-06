"""Public orchestrator entrypoint."""
from __future__ import annotations

from .orchestration import C, Orchestrator, clear_screen, fmt_time, main, progress_bar

__all__ = ["C", "Orchestrator", "clear_screen", "fmt_time", "main", "progress_bar"]

if __name__ == "__main__":
    main()
