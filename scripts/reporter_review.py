"""List and approve Reporter promotion requests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa.reporter_promotion import approve_promotion, pending_promotions
from ipa.reporter_store import ReporterStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Reporter reporter.db")
    parser.add_argument("--approve", default=None, help="Promotion ID to approve")
    parser.add_argument("--decided-by", default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    with ReporterStore(args.db) as store:
        if args.approve:
            if not args.decided_by:
                parser.error("--decided-by is required with --approve")
            approve_promotion(store, args.approve, args.decided_by, args.note)
            print(f"Approved: {args.approve}")
        else:
            print(json.dumps(pending_promotions(store), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
