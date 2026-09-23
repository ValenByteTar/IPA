"""Deterministic, read-only repository for IPA Engineering Knowledge System."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
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
ARTIFACT_RE = re.compile(r"\boutputs/[\w.\-/]+")
DOCS_RE = re.compile(r"\bdocs/[\w.\-/]+")

# Governance rules graduated by creation date: records authored from this
# day on must meet the promotion gate (evidence) and declare author_model
# when written by an agent. Older records get warnings, not errors.
GOVERNANCE_CUTOFF = date(2026, 9, 23)

# Records whose content stays searchable but that should not be offered as
# current guidance in context packages.
CLOSED_STATUSES = {"superseded", "rejected"}

# Governing records at which a path scope is a hot zone — likely collision
# point for parallel sessions (PAT-009). Shared by report() and work permits.
HOT_ZONE_THRESHOLD = 4

# Directory names skipped when listing repo files for `affects` liveness.
# Runtime/generated trees are excluded: `outputs/` alone holds ~160k files and
# `affects` globs under it are exempt from liveness anyway. `Landing/` stays
# (DEC-007 governs `Landing/**`).
_LIVENESS_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
                       ".pytest_cache", ".mypy_cache", ".ruff_cache",
                       "outputs", "Archive", "Transit", "models",
                       "local_archive", "build", "dist", "exllamav3-dev"}


def glob_match(path: str, pattern: str) -> bool:
    """Gitignore-style glob on repo-relative posix paths.

    `**` crosses separators, `*` and `?` do not. fnmatch lets `*` match `/`
    and PurePath.match has edge cases with trailing `**`, so the glob is
    translated to an explicit regex.
    """
    pattern = pattern.replace("\\", "/").lstrip("/")
    path = path.replace("\\", "/").lstrip("/")
    out = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            if pattern[i:i + 2] == "**":
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = pattern.find("]", i + 1)
            out.append(pattern[i:end + 1] if end > i else re.escape(char))
            i = end if end > i else i
        else:
            out.append(re.escape(char))
        i += 1
    return re.fullmatch("".join(out), path) is not None


def scope_prefix(glob: str) -> str:
    """Literal directory prefix of a glob — the part before any wildcard."""
    glob = glob.replace("\\", "/").lstrip("/")
    for index, char in enumerate(glob):
        if char in "*?[":
            return glob[:index].rsplit("/", 1)[0] + "/" if "/" in glob[:index] else ""
    return glob.rsplit("/", 1)[0] + "/" if "/" in glob else ""


def scopes_overlap(first: str, second: str) -> bool:
    """Conservative glob-vs-glob overlap via literal prefixes.

    `src/ipa/**` overlaps `src/ipa/agentic/promotion*`; `src/**` does not
    overlap `docs/**`. May over-report on exotic patterns — safe direction.
    """
    a, b = scope_prefix(first), scope_prefix(second)
    return a.startswith(b) or b.startswith(a)

# Weighted token-overlap scoring: a term hit in the title or the metadata is
# a much stronger signal than a hit somewhere in the body.
TITLE_WEIGHT = 5
METADATA_WEIGHT = 3
BODY_WEIGHT = 1


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
    def metadata_text(self) -> str:
        parts = [self.relative_path]
        for key in ("id", "category", "status", "components", "tags", "related"):
            value = self.metadata.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts).casefold()

    @property
    def searchable_text(self) -> str:
        return " ".join([self.title, self.body, self.metadata_text])


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

    def _component_vocabulary(self) -> tuple[set[str], dict[str, str], dict[str, list[str]]] | None:
        """Load `knowledge/_schema/components.json` if present.

        Returns (canonical_names, alias→canonical, group→members). None when no
        vocabulary file exists — validation stays permissive on unknown
        components.
        """
        path = self.root / "_schema" / "components.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        canonical = {str(item) for item in data.get("components") or []}
        aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
        groups = {
            str(group): [str(item) for item in members if str(item).strip()]
            for group, members in (data.get("groups") or {}).items()
            if isinstance(members, list)
        }
        return canonical, aliases, groups

    def component_matches(self, record_components: Iterable[str], value: str) -> bool:
        """Does a record's component set satisfy the requested component?

        Resolves aliases and expands `groups` in both directions: asking for the
        generic group `indexes` matches a record tagged `vector_index`, and
        asking for `vector_index` matches one tagged with the generic group.
        Without a vocabulary file this degrades to exact membership.
        """
        if not isinstance(value, str) or not value.strip():
            return True
        record_set = {str(item).casefold() for item in record_components}
        wanted = value.strip().casefold()
        vocabulary = self._component_vocabulary()
        if vocabulary is None:
            return wanted in record_set
        _, aliases, groups = vocabulary
        wanted = aliases.get(wanted, wanted).casefold()
        if wanted in record_set:
            return True
        if any(str(member).casefold() in record_set for member in groups.get(wanted, [])):
            return True
        return any(group.casefold() in record_set
                   for group, members in groups.items()
                   if wanted in {str(member).casefold() for member in members})

    def _missing_artifact_links(self, record: EKSRecord) -> list[str]:
        """`outputs/...` paths cited in the body that no longer exist.

        Existence is checked against the repository parent (project root when
        root=knowledge/). Warnings only: PM-002 showed evidence rot is real,
        but generated output is not committed and CI must not fail on it.

        Three kinds of citation are not evidence rot and are skipped:
        runtime leases (`.lock`), runtime state under `outputs/agent/`
        (staging corpora, review queues — created on demand), and citations
        on a line that already declares the artifact historical (the BM/EXP
        records that reference PM-002). Everything else still warns.
        """
        base = self.root.parent
        historical_markers = ("histórico", "historico", "histórica", "historica",
                              "ya no existe", "ya no existen", "no disponible")
        missing = []
        for line in record.body.splitlines():
            if any(marker in line.casefold() for marker in historical_markers):
                continue
            for match in ARTIFACT_RE.findall(line):
                if match.endswith(".lock") or match.startswith("outputs/agent/"):
                    continue  # runtime-transient: lease or on-demand agent state
                candidate = (base / match).resolve()
                if not candidate.exists():
                    missing.append(match)
        return missing

    def _repo_files(self) -> list[str]:
        """Repo-relative posix paths, used for `affects` glob liveness."""
        base = self.root.parent
        files: list[str] = []
        stack = [base]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir():
                    if entry.name not in _LIVENESS_SKIP_DIRS:
                        stack.append(entry)
                else:
                    files.append(entry.relative_to(base).as_posix())
        return files

    def _affects_list(self, record: EKSRecord) -> list[str]:
        value = record.metadata.get("affects")
        return [item for item in value if isinstance(item, str) and item.strip()] \
            if isinstance(value, list) else []

    def governing(
        self,
        paths: Iterable[str],
        *,
        include_closed: bool = True,
        categories: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Records whose `affects` globs match any of the given repo paths.

        `include_closed` defaults to True on purpose: rejected and superseded
        records governing a file are the graveyard — they are exactly what
        stops an agent from re-trying an already-discarded approach.
        """
        wanted = {str(p).replace("\\", "/").lstrip("/") for p in paths
                  if isinstance(p, str) and p.strip()}
        allowed_categories = set(categories) if categories else None
        hits: list[dict[str, Any]] = []
        for record in self.records():
            if allowed_categories and record.category not in allowed_categories:
                continue
            if not include_closed and record.metadata.get("status") in CLOSED_STATUSES:
                continue
            matched = [glob for glob in self._affects_list(record)
                       if any(glob_match(path, glob) for path in wanted)]
            if matched:
                hits.append({**self._summary(record), "affects_matched": matched})
        return hits

    def governing_scope(self, scope: Iterable[str]) -> list[dict[str, Any]]:
        """Records whose `affects` overlap a set of *globs* (e.g. a work
        permit's scope). Same prefix-overlap semantics as scopes_overlap."""
        hits = []
        for record in self.records():
            matched = [g for g in self._affects_list(record)
                       if any(scopes_overlap(g, s) for s in scope)]
            if matched:
                hits.append({**self._summary(record), "affects_matched": matched})
        return hits

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
            created, updated = metadata.get("created"), metadata.get("updated")
            if (
                isinstance(created, str) and isinstance(updated, str)
                and DATE_RE.fullmatch(created) and DATE_RE.fullmatch(updated)
                and updated < created
            ):
                errors.append(f"{record.relative_path}: updated {updated} predates created {created}")
            for list_field in ("components", "tags", "related"):
                value = metadata.get(list_field)
                if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                    errors.append(f"{record.relative_path}: {list_field} must be a list of non-empty strings")
            for link_field in ("supersedes", "superseded_by"):
                value = metadata.get(link_field)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append(f"{record.relative_path}: {link_field} must be string or null")
            status_value = metadata.get("status")
            if status_value == "superseded" and not metadata.get("superseded_by"):
                errors.append(f"{record.relative_path}: superseded records must declare superseded_by")
            if metadata.get("superseded_by") and status_value != "superseded":
                warnings.append(f"{record.relative_path}: superseded_by set but status is {status_value}")
        vocabulary = self._component_vocabulary()
        if vocabulary is not None:
            canonical, aliases, _ = vocabulary
            for record in records:
                for component in record.metadata.get("components", []) if isinstance(record.metadata.get("components"), list) else []:
                    if component in aliases:
                        warnings.append(
                            f"{record.relative_path}: component '{component}' is an alias of '{aliases[component]}'")
                    elif component not in canonical:
                        warnings.append(f"{record.relative_path}: unknown component '{component}'")
        for record in records:
            for field_name in ("supersedes", "superseded_by"):
                link = record.metadata.get(field_name)
                if isinstance(link, str) and ID_RE.fullmatch(link) and link not in seen:
                    errors.append(f"{record.relative_path}: {field_name} references missing {link}")
            for link in record.metadata.get("related", []) if isinstance(record.metadata.get("related"), list) else []:
                if ID_RE.fullmatch(link) and link not in seen:
                    warnings.append(f"{record.relative_path}: related references missing {link}")
        by_id = {record.record_id: record for record in records if record.record_id}
        for record in records:
            target_id = record.metadata.get("supersedes")
            target = by_id.get(target_id) if isinstance(target_id, str) else None
            if target is not None:
                if target.metadata.get("status") != "superseded":
                    warnings.append(
                        f"{record.relative_path}: supersedes {target_id} but its status is "
                        f"{target.metadata.get('status')}")
                if target.metadata.get("superseded_by") != record.record_id:
                    warnings.append(
                        f"{record.relative_path}: supersedes {target_id} but that record does not "
                        f"declare superseded_by {record.record_id}")
            successor_id = record.metadata.get("superseded_by")
            successor = by_id.get(successor_id) if isinstance(successor_id, str) else None
            if successor is not None and successor.metadata.get("supersedes") != record.record_id:
                warnings.append(
                    f"{record.relative_path}: superseded_by {successor_id} but that record does not "
                    f"declare supersedes {record.record_id}")
            for link in self._missing_artifact_links(record):
                warnings.append(f"{record.relative_path}: cited artifact missing on disk: {link}")

        # --- Governance rules (graduated by GOVERNANCE_CUTOFF) -------------
        base = self.root.parent
        repo_files: list[str] | None = None
        legacy_without_evidence: list[str] = []
        for record in records:
            metadata = record.metadata
            created = metadata.get("created")
            try:
                is_new = isinstance(created, str) and date.fromisoformat(created) >= GOVERNANCE_CUTOFF
            except ValueError:
                is_new = False

            affects = metadata.get("affects")
            if affects is not None:
                if not isinstance(affects, list) or any(
                        not isinstance(item, str) or not item.strip() for item in affects):
                    errors.append(f"{record.relative_path}: affects must be a list of non-empty globs")
                else:
                    # Liveness: a glob matching nothing is a silently dead
                    # scope — same failure mode as PM-002's lost evidence.
                    # `outputs/` globs are exempt (runtime-transient paths).
                    if repo_files is None:
                        repo_files = self._repo_files()
                    for glob in affects:
                        if glob.startswith("outputs/"):
                            continue
                        if not any(glob_match(path, glob) for path in repo_files):
                            warnings.append(
                                f"{record.relative_path}: affects glob matches no files: {glob}")

            evidence = metadata.get("evidence")
            evidence_ok = False
            if evidence is not None:
                if not isinstance(evidence, list) or any(
                        not isinstance(item, str) or not item.strip() for item in evidence):
                    errors.append(f"{record.relative_path}: evidence must be a list of non-empty paths")
                else:
                    for entry in evidence:
                        if (base / entry).resolve().exists():
                            evidence_ok = True
                        else:
                            message = f"{record.relative_path}: evidence path missing on disk: {entry}"
                            (errors if is_new else warnings).append(message)
            if not evidence_ok:
                # Body citations to existing artifacts/docs also satisfy the
                # promotion gate — evidence is the explicit form.
                for match in [*ARTIFACT_RE.findall(record.body), *DOCS_RE.findall(record.body)]:
                    if not match.endswith(".lock") and (base / match).resolve().exists():
                        evidence_ok = True
                        break
            if metadata.get("status") == "accepted" and not evidence_ok:
                if is_new:
                    errors.append(
                        f"{record.relative_path}: accepted record has no verifiable evidence "
                        f"(evidence field or existing cited path)")
                else:
                    legacy_without_evidence.append(record.relative_path)
            if metadata.get("author") == "agent" and is_new and not metadata.get("author_model"):
                warnings.append(
                    f"{record.relative_path}: author 'agent' should declare author_model")
        if legacy_without_evidence:
            warnings.append(
                f"{len(legacy_without_evidence)} legacy accepted records lack verifiable "
                f"evidence (pre-governance): {', '.join(sorted(legacy_without_evidence))}")
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

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(TOKEN_RE.findall(text.casefold()))

    def _score(self, record: EKSRecord, terms: set[str]) -> tuple[int, list[str]]:
        """Weighted token overlap: title > metadata > body. Deterministic."""
        hits: list[str] = []
        score = 0
        title_hits = len(terms & self._token_set(record.title))
        if title_hits:
            score += TITLE_WEIGHT * title_hits
            hits.append("title")
        meta_hits = len(terms & self._token_set(record.metadata_text))
        if meta_hits:
            score += METADATA_WEIGHT * meta_hits
            hits.append("metadata")
        body_hits = len(terms & self._token_set(record.body))
        if body_hits:
            score += BODY_WEIGHT * body_hits
            hits.append("body")
        return score, hits

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        status: str | None = None,
        component: str | None = None,
        tag: str | None = None,
        components: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        limit: int = 10,
        include_historical: bool = True,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        terms = set(TOKEN_RE.findall(query.casefold()))
        required_components = {str(v).casefold() for v in components or () if isinstance(v, str) and v.strip()}
        if isinstance(component, str) and component.strip():
            required_components.add(component.casefold())
        required_tags = {str(v).casefold() for v in tags or () if isinstance(v, str) and v.strip()}
        if isinstance(tag, str) and tag.strip():
            required_tags.add(tag.casefold())
        scored: list[tuple[int, EKSRecord, list[str]]] = []
        for record in self.all_documents():
            if category and record.category != category:
                continue
            if status and record.metadata.get("status") != status:
                continue
            if not include_historical and record.metadata.get("status") in CLOSED_STATUSES:
                continue
            record_components = {str(item).casefold() for item in record.metadata.get("components", [])} if isinstance(record.metadata.get("components"), list) else set()
            record_tags = {str(item).casefold() for item in record.metadata.get("tags", [])} if isinstance(record.metadata.get("tags"), list) else set()
            if required_components and not all(
                    self.component_matches(record_components, value)
                    for value in required_components):
                continue
            if required_tags and not required_tags <= record_tags:
                continue
            score, hits = self._score(record, terms)
            if score:
                scored.append((score, record, hits))
        scored.sort(key=lambda item: (-item[0], item[1].relative_path.casefold()))
        return [
            {**self._summary(record), "score": score, "matched_fields": hits,
             "match_reason": "weighted_token_overlap"}
            for score, record, hits in scored[:limit]
        ]

    def context(
        self,
        task: str,
        *,
        components: Iterable[str] = (),
        tags: Iterable[str] = (),
        paths: Iterable[str] = (),
        limit: int = 8,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        component_values = [value for value in components if isinstance(value, str) and value.strip()]
        tag_values = [value for value in tags if isinstance(value, str) and value.strip()]
        path_values = [value for value in paths if isinstance(value, str) and value.strip()]
        query = " ".join([task, *component_values, *tag_values])
        results: list[dict[str, Any]] = []
        filters_relaxed = False
        if query.strip():
            results = self.search(
                query,
                components=component_values,
                tags=tag_values,
                limit=limit,
                include_historical=include_historical,
            )
            if not results and (component_values or tag_values):
                # Strict intersection may starve the package; retry unfiltered
                # and flag it so the caller knows the filters did not hold.
                filters_relaxed = True
                results = self.search(query, limit=limit, include_historical=include_historical)

        # 1-hop expansion: records linked via `related` from the hits give the
        # caller the decision/benchmark context behind each result.
        by_id = {record.record_id: record for record in self.all_documents() if record.record_id}
        seen = {item["id"] for item in results}
        related_items: list[dict[str, Any]] = []
        for item in results:
            source = by_id.get(item["id"])
            if source is None:
                continue
            for link in source.metadata.get("related", []) if isinstance(source.metadata.get("related"), list) else []:
                target = by_id.get(link)
                if target is None or link in seen:
                    continue
                if not include_historical and target.metadata.get("status") in CLOSED_STATUSES:
                    continue
                seen.add(link)
                related_items.append({**self._summary(target), "via": item["id"], "hop": 1})

        # File-scoped applicability: records whose `affects` globs cover the
        # paths being touched — activated by *where you edit*, not by asking.
        # Closed records included: the graveyard prevents re-tries.
        applicable = [item for item in self.governing(path_values) if item["id"] not in seen] \
            if path_values else []

        available = bool(results or related_items or applicable)
        return {
            "task": task,
            "results": results,
            "related": related_items,
            "applicable": applicable,
            "filters_relaxed": filters_relaxed,
            "knowledge_available": available,
            "notice": None if available else "No applicable EKS knowledge found in the local repository.",
        }

    def report(self, *, today: date | None = None) -> dict[str, Any]:
        """Hygiene report: lifecycle, coverage and link health.

        Complements `validate()` (form) with content-level signals: aging
        proposals, unreferenced records, component coverage and citations to
        artifacts that no longer exist on disk.
        """
        today = today or date.today()
        records = self.records()
        by_id = {r.record_id: r for r in records if r.record_id}
        inbound: dict[str, int] = {record_id: 0 for record_id in by_id}
        for record in records:
            for link in record.metadata.get("related", []) if isinstance(record.metadata.get("related"), list) else []:
                if link in inbound:
                    inbound[link] += 1

        def _age(value: Any) -> int | None:
            if isinstance(value, str) and DATE_RE.fullmatch(value):
                try:
                    return (today - date.fromisoformat(value)).days
                except ValueError:
                    return None
            return None

        base = self.root.parent

        def _has_evidence(record: EKSRecord) -> bool:
            evidence = record.metadata.get("evidence")
            if isinstance(evidence, list) and any(
                    isinstance(e, str) and (base / e).resolve().exists() for e in evidence):
                return True
            return any(
                not m.endswith(".lock") and (base / m).resolve().exists()
                for m in [*ARTIFACT_RE.findall(record.body), *DOCS_RE.findall(record.body)])

        open_items = [
            {"id": r.record_id, "title": r.title,
             "status": r.metadata.get("status"),
             "age_days": _age(r.metadata.get("created")),
             "has_evidence": _has_evidence(r)}
            for r in records
            if r.metadata.get("status") in {"draft", "proposed"}
        ]
        open_items.sort(key=lambda item: (-(item["age_days"] or 0), item["id"] or ""))

        coverage: dict[str, int] = {}
        for record in records:
            for component in record.metadata.get("components", []) if isinstance(record.metadata.get("components"), list) else []:
                coverage[component] = coverage.get(component, 0) + 1

        missing_artifacts = {
            record.record_id: sorted(set(self._missing_artifact_links(record)))
            for record in records
        }
        missing_artifacts = {key: value for key, value in missing_artifacts.items() if value}

        # Hot zones: path scopes governed by many records — likely collision
        # points for parallel sessions.
        zone_hits: dict[str, list[str]] = {}
        for record in records:
            for glob in self._affects_list(record):
                zone_hits.setdefault(glob, []).append(record.record_id or record.relative_path)
        hot_zones = {glob: ids for glob, ids in zone_hits.items() if len(ids) >= HOT_ZONE_THRESHOLD}

        # Overlap view: the same prefix criterion `governing_scope` uses when a
        # work permit is issued, so the report and the precautions a session
        # actually receives cannot disagree.
        candidate_scopes = sorted({
            prefix for record in records for glob in self._affects_list(record)
            if (prefix := scope_prefix(glob))
        })
        hot_zones_overlap = {
            scope: sorted(ids) for scope in candidate_scopes
            if len(ids := [r.record_id or r.relative_path for r in records
                           if any(scopes_overlap(glob, scope)
                                  for glob in self._affects_list(r))]) >= HOT_ZONE_THRESHOLD
        }

        author_models: dict[str, int] = {}
        for record in records:
            model = record.metadata.get("author_model")
            if isinstance(model, str) and model.strip():
                author_models[model] = author_models.get(model, 0) + 1

        return {
            "total_records": len(records),
            "by_category": {cat: sum(1 for r in records if r.category == cat) for cat in CATEGORIES},
            "by_status": {status: sum(1 for r in records if r.metadata.get("status") == status)
                          for status in ("draft", "proposed", "accepted", "rejected", "superseded")},
            "open_items": open_items,
            "awaiting_evidence": [item["id"] for item in open_items
                                  if not item["has_evidence"] and (item["age_days"] or 0) > 7],
            "unreferenced": sorted(record_id for record_id, count in inbound.items() if count == 0),
            "component_coverage": dict(sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0]))),
            "missing_artifacts": missing_artifacts,
            "hot_zones": hot_zones,
            "hot_zones_overlap": hot_zones_overlap,
            "author_models": dict(sorted(author_models.items(), key=lambda kv: (-kv[1], kv[0]))),
            "reference_documents": len(self.references()),
        }

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
