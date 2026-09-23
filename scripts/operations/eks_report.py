"""Print a hygiene report over the EKS catalog.

Complements validate_eks.py (form) with content signals: aging proposals,
unreferenced records, component coverage and citations to artifacts that no
longer exist on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from eks_repository import EKSRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="knowledge")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = EKSRepository(Path(args.root)).report()
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"EKS records: {report['total_records']} "
          f"({report['reference_documents']} reference docs)")
    print("by status:  " + ", ".join(
        f"{k}={v}" for k, v in report["by_status"].items() if v))
    print("by category: " + ", ".join(
        f"{k}={v}" for k, v in report["by_category"].items() if v))

    if report["open_items"]:
        print("\nOpen items (draft/proposed):")
        for item in report["open_items"]:
            age = f"{item['age_days']}d" if item["age_days"] is not None else "?"
            ev = "" if item.get("has_evidence") else " — sin evidencia"
            print(f"  {item['id']} [{item['status']}, {age}]{ev} {item['title']}")

    if report.get("awaiting_evidence"):
        print("\nAwaiting evidence (>7d open, none verifiable):")
        for record_id in report["awaiting_evidence"]:
            print(f"  {record_id}")

    if report["unreferenced"]:
        print("\nUnreferenced (no inbound `related`):")
        for record_id in report["unreferenced"]:
            print(f"  {record_id}")

    if report["missing_artifacts"]:
        print("\nCited artifacts missing on disk:")
        for record_id, links in report["missing_artifacts"].items():
            for link in links:
                print(f"  {record_id}: {link}")

    if report.get("hot_zones"):
        print("\nHot zones — exact glob (>=4 records govern the same glob):")
        for glob, ids in report["hot_zones"].items():
            print(f"  {glob}: {', '.join(ids)}")

    if report.get("hot_zones_overlap"):
        print("\nHot zones — prefix overlap (same criterion as permit precautions):")
        for scope, ids in report["hot_zones_overlap"].items():
            print(f"  {scope}: {', '.join(ids)}")

    if report.get("author_models"):
        print("\nAuthor models:")
        for model, count in report["author_models"].items():
            print(f"  {model}: {count}")

    print("\nComponent coverage:")
    for component, count in report["component_coverage"].items():
        print(f"  {component}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
