"""IPA Orchestrator — root-level compatibility entrypoint.

Delegates to the canonical implementation in ipa.dashboard.orchestrator.
Prefer scripts/operations/orchestrator.py for new usage.

Usage:
    python scripts/orchestrator.py                          # full system
    python scripts/orchestrator.py --no-scraper --no-hammer # pipeline + lancedb + enrichment
    python scripts/orchestrator.py --chunker semantic       # use semantic chunker
"""
import sys
from pathlib import Path

# Ensure src/ is on the path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ipa.dashboard.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    main()
