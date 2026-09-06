"""Deterministic, read-only repository for IPA Engineering Knowledge System."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

CATEGORIES = {
    "decision": ("decisions", "DEC"),
    "experiment": ("experiments", "EXP"),
    "benchmark": ("benchmarks", "BM"),
    "postmortem": ("postmortems", "PM"),
    "pattern": ("patterns", "PAT"),
    "research": ("research", "RES"),
}
PREFIX_TO_CATEGORY = {prefix: category for category, (_, prefix) in CATEGORIES.items()}
ID_RE = re.compile(r"^(DEC|EXP|BM|PM|PAT|RES)-\d{3,}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True)
class EKSRecord:
    path: Path
    relative_path: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    reference: bool = False

    @property
    def record_id(self) -> str | None:
        value = self.metadata.get("id")
        return value if isinstance(value, str) else None

    @property
    def category(self) -> str:
        value = self.metadata.get("category")
        return value if isinstance(value, str) else "reference"

    @property
    def searchable_text(self) -> str:
        parts = [self.title, self.body, self.relative_path]
        for key in ("id", "category", "status", "components", "tags", "related"):
            value = self.metadata.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts).casefold()


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    records: tuple[EKSRecord, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


class EKSRepository:
    """Read-only EKS catalog with bounded roots and stable ordering."""

    def __init__(self, root: str | Path, reference_roots: Iterable[str | Path] = ()) -> None:
        self.root = Path(root).expanduser().resolve()
        self.reference_roots = tuple(Path(path).expanduser().resolve() for path in reference_roots)

    def _inside(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            return False

    def _read_markdown(self, path: Path, *, reference: bool = False) -> EKSRecord:
        text = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        body = text
        if text.startswith("---\n"):
            end = text.find("\n---", 5)
            if end >= 0:
                raw = text[4:end]
                loaded = yaml.safe_load(raw) or {}
                if isinstance(loaded, dict):
                    for date_field in ("created", "updated"):
                        value = loaded.get(date_field)
                        if hasattr(value, "isoformat") and not isinstance(value, str):
                            loaded[date_field] = value.isoformat()
                    metadata = loaded
                else:
                    metadata = {"_frontmatter_error": "not a mapping"}
                body = text[end + 5 :].lstrip("\r\n")
        title = next((line[2:].strip() for line in body.splitlines() if line.startswith("# ")), path.stem)
        relative = self._relative(path)
        return EKSRecord(path.resolve(), relative, title, metadata, body, reference)

    def _relative(self, path: Path) -> str:
        path = path.resolve()
        if self._inside(path, self.root):
            return path.relative_to(self.root).as_posix()
        for ref_root in self.reference_roots:
            if self._inside(path, ref_root):
                return f"{ref_root.name}/{path.relative_to(ref_root).as_posix()}"
        raise ValueError("path outside repository roots")

    def records(self) -> list[EKSRecord]:
        records: list[EKSRecord] = []
        for category, (directory, _) in CATEGORIES.items():
            folder = self.root / directory
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*.md"), key=lambda item: item.as_posix().casefold()):
                records.append(self._read_markdown(path))
        return records

    def references(self) -> list[EKSRecord]:
        records: list[EKSRecord] = []
        for root in self.reference_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.md"), key=lambda item: item.as_posix().casefold()):
                records.append(self._read_markdown(path, reference=True))
        return records

    def all_documents(self) -> list[EKSRecord]:
        return [*self.records(), *self.references()]

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        records = self.records()
        seen: dict[str, Path] = {}
        for record in records:
            metadata = record.metadata
            prefix = record.path.parent.name
            if "_frontmatter_error" in metadata:
                errors.append(f"{record.relative_path}: frontmatter must be a YAML mapping")
                continue
            required = ("id", "category", "status", "created", "updated", "author", "components", "tags", "related", "supersedes", "superseded_by")
            for field_name in required:
                if field_name not in metadata:
                    errors.append(f"{record.relative_path}: missing metadata field {field_name}")
            record_id = metadata.get("id")
            category = metadata.get("category")
            if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
                errors.append(f"{record.relative_path}: invalid id")
            else:
                if record_id in seen:
                    errors.append(f"{record.relative_path}: duplicate id {record_id} (already in {seen[record_id]})")
                seen[record_id] = record.path
                expected_category = PREFIX_TO_CATEGORY[record_id.split("-", 1)[0]]
                if category != expected_category:
                    errors.append(f"{record.relative_path}: id/category mismatch")
            if category not in CATEGORIES:
                errors.append(f"{record.relative_path}: invalid category")
            elif prefix != CATEGORIES[category][0]:
                errors.append(f"{record.relative_path}: category folder mismatch")
            if metadata.get("status") not in {"draft", "proposed", "accepted", "rejected", "superseded"}:
                errors.append(f"{record.relative_path}: invalid status")
            for date_field in ("created", "updated"):
                value = metadata.get(date_field)
                if not isinstance(value, str) or not DATE_RE.fullmatch(value):
                    errors.append(f"{record.relative_path}: invalid {date_field}")
            for list_field in ("components", "tags", "related"):
                value = metadata.get(list_field)
                if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                    errors.append(f"{record.relative_path}: {list_field} must be a list of non-empty strings")
            for link_field in ("supersedes", "superseded_by"):
                value = metadata.get(link_field)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append(f"{record.relative_path}: {link_field} must be string or null")
        for record in records:
            for field_name in ("supersedes", "superseded_by"):
                link = record.metadata.get(field_name)
                if isinstance(link, str) and ID_RE.fullmatch(link) and link not in seen:
                    errors.append(f"{record.relative_path}: {field_name} references missing {link}")
            for link in record.metadata.get("related", []) if isinstance(record.metadata.get("related"), list) else []:
                if ID_RE.fullmatch(link) and link not in seen:
                    warnings.append(f"{record.relative_path}: related references missing {link}")
        return ValidationReport(tuple(sorted(set(errors))), tuple(sorted(set(warnings))), tuple(records))

    def get(self, identifier: str) -> EKSRecord:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be a non-empty string")
        value = identifier.strip().replace("\\", "/")
        for record in self.all_documents():
            if record.record_id == value or record.relative_path == value or record.relative_path == value.removeprefix("./"):
                return record
        candidate = (self.root / value).resolve()
        allowed = self._inside(candidate, self.root) or any(self._inside(candidate, root) for root in self.reference_roots)
        if allowed and candidate.is_file() and candidate.suffix.casefold() == ".md":
            return self._read_markdown(candidate, reference=not self._inside(candidate, self.root))
        raise KeyError(f"EKS document not found: {identifier}")

    def search(self, query: str, *, category: str | None = None, status: str | None = None, component: str | None = None, tag: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        terms = set(TOKEN_RE.findall(query.casefold()))
        scored: list[tuple[int, EKSRecord]] = []
        for record in self.all_documents():
            if category and record.category != category:
                continue
            if status and record.metadata.get("status") != status:
                continue
            components = {str(item).casefold() for item in record.metadata.get("components", [])} if isinstance(record.metadata.get("components"), list) else set()
            tags = {str(item).casefold() for item in record.metadata.get("tags", [])} if isinstance(record.metadata.get("tags"), list) else set()
            if component and component.casefold() not in components:
                continue
            if tag and tag.casefold() not in tags:
                continue
            searchable = record.searchable_text
            score = sum(1 for term in terms if term in searchable)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].relative_path.casefold()))
        return [{**self._summary(record), "score": score, "match_reason": "token_overlap"} for score, record in scored[:limit]]

    def context(self, task: str, *, components: Iterable[str] = (), tags: Iterable[str] = (), limit: int = 8) -> dict[str, Any]:
        component_values = [value for value in components if isinstance(value, str) and value.strip()]
        tag_values = [value for value in tags if isinstance(value, str) and value.strip()]
        query = " ".join([task, *component_values, *tag_values])
        results = self.search(query, component=component_values[0] if len(component_values) == 1 else None, tag=tag_values[0] if len(tag_values) == 1 else None, limit=limit) if query.strip() else []
        return {"task": task, "results": results, "knowledge_available": bool(results), "notice": None if results else "No applicable EKS knowledge found in the local repository."}

    @staticmethod
    def _summary(record: EKSRecord) -> dict[str, Any]:
        return {
            "id": record.record_id,
            "category": record.category,
            "status": record.metadata.get("status", "reference"),
            "title": record.title,
            "path": record.relative_path,
            "components": record.metadata.get("components", []),
            "tags": record.metadata.get("tags", []),
            "related": record.metadata.get("related", []),
            "reference": record.reference,
        }

    def serialize(self, record: EKSRecord) -> dict[str, Any]:
        return {**self._summary(record), "metadata": record.metadata, "content": record.body}
