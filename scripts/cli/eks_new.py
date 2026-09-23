"""Create a new EKS record from the local templates.

Example:
    eks_new.py decision --title "DEC-* is the ADR format" --status proposed \
        --components eks configuration --tags governance adr --related RES-003
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from eks_repository import CATEGORIES, EKSRepository  # noqa: E402
from eks_scaffold import scaffold  # noqa: E402


def _split(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=sorted(CATEGORIES))
    parser.add_argument("--title", required=True)
    parser.add_argument("--status", default="draft",
                        choices=["draft", "proposed", "accepted", "rejected", "superseded"])
    parser.add_argument("--author", default="agent")
    parser.add_argument("--components", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--related", nargs="*", default=None)
    parser.add_argument("--supersedes", default=None)
    parser.add_argument("--affects", nargs="*", default=None,
                        help="Repo-relative globs this record governs")
    parser.add_argument("--evidence", nargs="*", default=None,
                        help="Repo-relative paths that prove the record")
    parser.add_argument("--author-model", dest="author_model", default=None)
    parser.add_argument("--trigger", default=None,
                        help="What originated the record (e.g. permit:PW-...)")
    parser.add_argument("--root", default="knowledge")
    args = parser.parse_args()

    root = Path(args.root)
    path = scaffold(
        root,
        args.category,
        args.title,
        status=args.status,
        author=args.author,
        components=_split(args.components),
        tags=_split(args.tags),
        related=_split(args.related),
        supersedes=args.supersedes,
        affects=_split(args.affects),
        evidence=_split(args.evidence),
        author_model=args.author_model,
        trigger=args.trigger,
    )
    report = EKSRepository(root).validate()
    print(f"created: {path}")
    for error in report.errors:
        print(f"ERROR: {error}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
