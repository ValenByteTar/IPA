"""Scaffold helpers for creating new EKS records from the local templates.

Write path lives outside the MCP server on purpose: the EKS MCP is read-only
by design (RES-003). This module is the sanctioned way to author a record —
it assigns the next id, renders the frontmatter and keeps the file inside the
category folder that matches the id prefix.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Iterable

try:
    from .eks_repository import CATEGORIES, ID_RE
except ImportError:  # Supports direct sys.path execution by scripts/tests.
    from eks_repository import CATEGORIES, ID_RE

FRONTMATTER_KEYS = (
    "id", "category", "status", "created", "updated", "author",
    "components", "tags", "related", "supersedes", "superseded_by",
    "affects", "evidence", "author_model", "trigger",
)


def _existing_ids(root: Path, prefix: str) -> list[int]:
    directory = root / next(folder for cat, (folder, p) in CATEGORIES.items() if p == prefix)
    numbers = []
    if directory.is_dir():
        for path in directory.rglob("*.md"):
            match = re.match(rf"{prefix}-(\d+)", path.stem)
            if match:
                numbers.append(int(match.group(1)))
    return numbers


def next_id(root: Path, category: str) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category}")
    _, prefix = CATEGORIES[category]
    following = max(_existing_ids(root, prefix), default=0) + 1
    return f"{prefix}-{following:03d}"


def slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return slug[:60].strip("-") or "untitled"


def _yaml_list(values: Iterable[str]) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    return "[" + ", ".join(items) + "]"


def render_frontmatter(
    record_id: str,
    category: str,
    *,
    status: str = "draft",
    author: str = "agent",
    components: Iterable[str] = (),
    tags: Iterable[str] = (),
    related: Iterable[str] = (),
    supersedes: str | None = None,
    affects: Iterable[str] = (),
    evidence: Iterable[str] = (),
    author_model: str | None = None,
    trigger: str | None = None,
    today: date | None = None,
) -> str:
    today = today or date.today()
    stamp = today.isoformat()
    lines = [
        "---",
        f"id: {record_id}",
        f"category: {category}",
        f"status: {status}",
        f"created: {stamp}",
        f"updated: {stamp}",
        f"author: {author}",
        f"components: {_yaml_list(components)}",
        f"tags: {_yaml_list(tags)}",
        f"related: {_yaml_list(related)}",
        f"supersedes: {supersedes}" if supersedes else "supersedes: null",
        "superseded_by: null",
        f"affects: {_yaml_list(affects)}",
        f"evidence: {_yaml_list(evidence)}",
        f"author_model: {author_model}" if author_model else "author_model: null",
        f"trigger: {trigger}" if trigger else "trigger: null",
        "---",
    ]
    return "\n".join(lines)


def _render_body(root: Path, category: str, record_id: str, title: str) -> str:
    template = root / "_templates" / f"{category}.md"
    if template.is_file():
        text = template.read_text(encoding="utf-8")
        end = text.find("\n---", 4)
        body = text[end + 5:] if text.startswith("---\n") and end >= 0 else text
        # Replace the template's placeholder H1 ("# DEC-001 — Título").
        lines = body.lstrip("\r\n").splitlines()
        if lines and lines[0].startswith("# "):
            lines[0] = f"# {record_id} — {title}"
            return "\n".join(lines)
        return f"# {record_id} — {title}\n\n" + body.lstrip("\r\n")
    return f"# {record_id} — {title}\n"


def scaffold(
    root: Path,
    category: str,
    title: str,
    *,
    status: str = "draft",
    author: str = "agent",
    components: Iterable[str] = (),
    tags: Iterable[str] = (),
    related: Iterable[str] = (),
    supersedes: str | None = None,
    affects: Iterable[str] = (),
    evidence: Iterable[str] = (),
    author_model: str | None = None,
    trigger: str | None = None,
    today: date | None = None,
) -> Path:
    """Create a new EKS record file and return its path."""
    if not title or not title.strip():
        raise ValueError("title must be non-empty")
    if status not in {"draft", "proposed", "accepted", "rejected", "superseded"}:
        raise ValueError(f"invalid status: {status}")
    if supersedes and not ID_RE.fullmatch(supersedes):
        raise ValueError(f"supersedes must be an EKS id, got: {supersedes}")
    record_id = next_id(root, category)
    folder, _ = CATEGORIES[category]
    path = root / folder / f"{record_id}-{slugify(title)}.md"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_frontmatter(
            record_id, category, status=status, author=author,
            components=components, tags=tags, related=related,
            supersedes=supersedes, affects=affects, evidence=evidence,
            author_model=author_model, trigger=trigger, today=today,
        )
        + "\n\n"
        + _render_body(root, category, record_id, title.strip())
    )
    path.write_text(content, encoding="utf-8")
    return path
