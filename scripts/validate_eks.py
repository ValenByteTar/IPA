"""Validate IPA EKS metadata without modifying the repository."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from eks_repository import EKSRepository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="knowledge")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    repository = EKSRepository(Path(args.root), [Path("docs/adr")])
    report = repository.validate()
    payload = {
        "valid": report.valid,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "records": len(report.records),
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"EKS valid: {report.valid}; records: {len(report.records)}")
        for error in report.errors:
            print(f"ERROR: {error}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
