"""Landing sweep — enforce Landing as a transit zone, not permanent storage.

Policy (see docs/operations/landing-and-archive.md):

- processed + approved (in main corpus)      → ``Archive/``
- processed + pending human confirmation     → ``Transit/``
- processed + rejected                       → delete
- unprocessed / in-flight / unregistered     → stays in ``Landing/``

"Approved" means a non-tombstoned document with that artifact_id exists in the
MAIN corpus document_store. "Pending" means the artifact was ingested into a
staging corpus but is not in main yet (promotion queue or curation review
pending), or a registry marked it ``indexed`` without store confirmation.
"Rejected" means every registry that knows the artifact reports ``failed`` and
no store contains it, or a human set its curation ``review_status='rejected'``.

The sweep also re-scans ``Transit/``: files whose document was promoted to the
main corpus since the last sweep move on to ``Archive/``; files whose document
was rejected are deleted.

Operational files (``*.db``, ``*.pending_delete``, hidden files) are never
touched — only artifacts registered in a landing.db are considered, and only
files that still live under ``landing_root`` / ``transit_root`` are acted on.
"""
from __future__ import annotations

import gc
import json
import shutil
import sqlite3
import time
from pathlib import Path

from ipa.ingestion.landing_zone import _sha256

_STATUS_RANK = {
    "failed": 0,
    "received": 1,
    "accepted": 2,
    "parsing": 3,
    "quarantine": 3,
    "chunked": 4,
    "indexed": 5,
    "no_text": 5,
}

# Operational scraper state living inside Landing — never artifacts to sweep.
_OPERATIONAL_NAMES = {"scrape_report.json", "scrape_history.db"}


def _is_operational(path: Path) -> bool:
    return (path.name in _OPERATIONAL_NAMES or path.name.startswith(".")
            or path.suffix == ".db" or path.suffix == ".pending_delete")


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_artifacts(landing_db: Path) -> dict[str, dict[str, str]]:
    """artifact_id -> {status, source_uri} from one corpus landing.db."""
    if not landing_db.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with sqlite3.connect(str(landing_db), timeout=30.0) as conn:
            for aid, status, uri in conn.execute(
                "SELECT artifact_id, status, source_uri FROM artifacts"
            ):
                out[str(aid)] = {"status": str(status or ""), "source_uri": str(uri or "")}
    except sqlite3.Error:
        pass
    return out


def _live_docs(document_store: Path) -> dict[str, str]:
    """artifact_id -> document_id for live (non-tombstoned) documents."""
    if not document_store.exists():
        return {}
    try:
        with sqlite3.connect(str(document_store), timeout=30.0) as conn:
            rows = conn.execute(
                "SELECT DISTINCT artifact_id, document_id FROM documents WHERE tombstoned = 0"
            ).fetchall()
        return {str(a): str(d) for a, d in rows}
    except sqlite3.Error:
        return {}


def _review_states(cluster_db: Path, reporter_db: Path | None = None) -> tuple[set[str], set[str]]:
    """(rejected_doc_ids, pending_doc_ids) from curation and review decisions.

    Two decision stores feed the signal:

    - ``topic_clusters.db``: ``curation_decisions.payload_json`` (idle
      enrichment) and ``promotion_queue`` (policy evaluation).
    - ``reporter.db``: ``document_decisions.payload_json`` — this is where the
      dashboard's ``/api/decisions/review`` writes human approve/reject.

    Both hold a ``review_status`` field (pending|approved|rejected|
    changes_requested). Rows come back newest-first; first status seen per
    document wins.
    """
    rejected: set[str] = set()
    pending: set[str] = set()

    def _read_payloads(conn: sqlite3.Connection, table: str, doc_col: str | None) -> None:
        seen: set[str] = set()
        col = f"{doc_col}, payload_json" if doc_col else "payload_json"
        for row in conn.execute(f"SELECT {col} FROM {table} ORDER BY created_at DESC"):
            try:
                d = json.loads(row[-1])
            except (TypeError, json.JSONDecodeError):
                continue
            doc_id = str(row[0] if doc_col else d.get("document_id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            status = str(d.get("review_status") or "pending")
            if status == "rejected":
                rejected.add(doc_id)
            elif status != "approved":
                pending.add(doc_id)

    for db, table, doc_col in (
        (cluster_db, "curation_decisions", None),
        (reporter_db, "document_decisions", "document_id"),
    ):
        if db is None or not Path(db).exists():
            continue
        try:
            with sqlite3.connect(str(db), timeout=30.0) as conn:
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                if table in tables:
                    _read_payloads(conn, table, doc_col)
                if "promotion_queue" in tables:
                    for (doc_id,) in conn.execute(
                        "SELECT document_id FROM promotion_queue WHERE status = 'pending'"
                    ):
                        pending.add(str(doc_id))
        except sqlite3.Error:
            pass
    # A human reject is the strongest negative signal: it wins over pending.
    return rejected, pending - rejected


def _robust_delete(path: Path) -> None:
    """Delete a file, surviving transient Windows file locks.

    On persistent lock the file is renamed to ``.pending_delete`` so the next
    sweep (or pipeline startup) removes it once handles are released.
    """
    gc.collect()
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            if not path.exists():
                return
        except (PermissionError, OSError):
            if attempt < 4:
                time.sleep(0.5 * (2 ** attempt))
                gc.collect()
            else:
                try:
                    path.rename(path.with_suffix(path.suffix + ".pending_delete"))
                except (PermissionError, OSError):
                    pass
                return


def _move_to_dir(path: Path, landing_root: Path, dest_root: Path,
                 existing_hashes: dict[str, Path] | None = None) -> Path:
    """Copy file into dest_root preserving the Landing-relative path, then delete.

    Dedup by content: if ``existing_hashes`` already contains the file's hash
    (same content anywhere in dest_root) or a file with the same content sits
    at the destination path, the source is just deleted — the scraper can
    re-download identical artifacts under different names.
    """
    try:
        rel = path.resolve().relative_to(landing_root.resolve())
    except ValueError:
        rel = Path(path.name)
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    gc.collect()
    try:
        file_hash = "sha256:" + _sha256(path)
    except OSError:
        file_hash = None
    if existing_hashes is not None and file_hash and file_hash in existing_hashes:
        _robust_delete(path)
        return existing_hashes[file_hash]
    if dest.exists():
        try:
            if file_hash and "sha256:" + _sha256(dest) == file_hash:
                _robust_delete(path)
                return dest
        except OSError:
            pass
        dest = dest.with_name(f"{dest.stem}_{int(time.time()) % 100000}{dest.suffix}")
    shutil.copy2(str(path), str(dest))
    if existing_hashes is not None and file_hash:
        existing_hashes[file_hash] = dest
    _robust_delete(path)
    return dest


def cleanup_pending_delete(*roots: Path) -> int:
    """Remove ``*.pending_delete`` leftovers from previous sweeps."""
    cleaned = 0
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*.pending_delete"):
            try:
                f.unlink(missing_ok=True)
                cleaned += 1
            except (PermissionError, OSError):
                pass
    return cleaned


def sweep_landing(
    landing_root: str | Path,
    archive_root: str | Path,
    corpora: list[str | Path],
    *,
    main_corpus: str | Path | None = None,
    transit_root: str | Path | None = None,
    cluster_db: str | Path | None = None,
    reporter_db: str | Path | None = None,
) -> dict:
    """Enforce the Landing transit policy.

    ``corpora`` are corpus directories that may contain ``landing.db`` (artifact
    registry) and ``document_store.db`` (canonical store). ``main_corpus`` is
    the authoritative corpus: membership there means approved → Archive.
    ``transit_root`` (default ``Landing/../Transit``) receives processed
    artifacts still awaiting confirmation into the main corpus. ``cluster_db``
    (topic_clusters.db) supplies curation review_status and the promotion
    queue; ``reporter_db`` supplies the dashboard's human review decisions.
    """
    landing_root = Path(landing_root)
    archive_root = Path(archive_root)
    main_path = Path(main_corpus) if main_corpus else None
    transit_root = Path(transit_root) if transit_root else landing_root.parent / "Transit"
    stats = {"archived": 0, "transit": 0, "deleted": 0, "skipped": 0,
             "pending_cleaned": 0, "errors": []}

    # Union of registries: highest-rank status wins (indexed > failed).
    artifacts: dict[str, dict[str, str]] = {}
    main_ids: set[str] = set()          # artifact_ids live in the main corpus
    staged_ids: set[str] = set()        # artifact_ids live in non-main corpora
    doc_to_artifact: dict[str, str] = {}
    for corpus in corpora:
        corpus = Path(corpus)
        for aid, info in _load_artifacts(corpus / "landing.db").items():
            prev = artifacts.get(aid)
            if prev is None or _STATUS_RANK.get(info["status"], -1) > _STATUS_RANK.get(prev["status"], -1):
                artifacts[aid] = info
        docs = _live_docs(corpus / "document_store.db")
        for aid, doc_id in docs.items():
            doc_to_artifact[doc_id] = aid
        if main_path is not None and corpus.resolve() == main_path.resolve():
            main_ids |= set(docs)
        else:
            staged_ids |= set(docs)
    # When no main corpus is given, any corpus membership counts as approved.
    if main_path is None:
        main_ids |= staged_ids
        staged_ids = set()

    # Human-review and promotion-queue signals, keyed by document_id.
    rejected_docs, pending_docs = _review_states(
        Path(cluster_db) if cluster_db else Path("__missing__"),
        Path(reporter_db) if reporter_db else None)
    rejected_artifacts = {doc_to_artifact[d] for d in rejected_docs if d in doc_to_artifact}
    pending_artifacts = {doc_to_artifact[d] for d in pending_docs if d in doc_to_artifact}
    pending_artifacts |= staged_ids - main_ids

    # Content index of Transit — dedup target for incoming moves and the
    # lookup table for the Transit re-evaluation pass.
    transit_hashes: dict[str, Path] = {}
    if transit_root.exists():
        for path in sorted(transit_root.rglob("*")):
            if not path.is_file() or _is_operational(path):
                continue
            try:
                transit_hashes["sha256:" + _sha256(path)] = path
            except OSError:
                continue
    rejected_artifacts = {doc_to_artifact[d] for d in rejected_docs if d in doc_to_artifact}
    pending_artifacts = {doc_to_artifact[d] for d in pending_docs if d in doc_to_artifact}
    pending_artifacts |= staged_ids - main_ids

    def _classify(aid: str, status: str) -> str:
        if aid in main_ids:
            return "archive"
        if aid in rejected_artifacts:
            return "delete"
        if aid in pending_artifacts or status == "indexed":
            return "transit"
        # no_text: parsed OK but zero usable text (e.g. image-only PDF whose
        # OCR produced nothing) — nothing queryable was ever stored, so the
        # file is waste like a failed parse.
        if status in ("failed", "no_text"):
            return "delete"
        return "skip"

    swept_paths: set[Path] = set()
    for aid, info in artifacts.items():
        source_uri = info["source_uri"]
        status = info["status"]
        if not source_uri:
            continue
        path = Path(source_uri)
        swept_paths.add(path.resolve())
        if not path.exists() or not path.is_file():
            continue  # already gone — nothing to sweep
        if not _under(path, landing_root):
            continue  # registered from a different root — not ours to move
        if _is_operational(path):
            continue  # operational state, never an artifact to sweep
        try:
            action = _classify(aid, status)
            if action == "archive":
                _move_to_dir(path, landing_root, archive_root)
                stats["archived"] += 1
            elif action == "transit":
                _move_to_dir(path, landing_root, transit_root, transit_hashes)
                stats["transit"] += 1
            elif action == "delete":
                _robust_delete(path)
                stats["deleted"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["errors"].append(f"{path.name}: {exc}")

    # Second pass, content-addressed: the scraper can re-download a file to a
    # path that is NOT the registered source_uri (register() is idempotent by
    # content hash and keeps the first path). Resolve those leftovers by hash.
    statuses_by_id = {aid: info["status"] for aid, info in artifacts.items()}
    for path in sorted(landing_root.rglob("*")):
        if not path.is_file():
            continue
        if _is_operational(path):
            continue
        if path.resolve() in swept_paths:
            continue
        try:
            aid = "sha256:" + _sha256(path)
        except OSError:
            continue
        try:
            action = _classify(aid, statuses_by_id.get(aid, ""))
            if action == "archive":
                _move_to_dir(path, landing_root, archive_root)
                stats["archived"] += 1
            elif action == "transit":
                _move_to_dir(path, landing_root, transit_root, transit_hashes)
                stats["transit"] += 1
            elif action == "delete":
                _robust_delete(path)
                stats["deleted"] += 1
            else:
                stats["skipped"] += 1  # unknown content — never seen, stays
        except Exception as exc:
            stats["errors"].append(f"{path.name}: {exc}")

    # Third pass: re-evaluate Transit. Promoted → Archive, rejected → delete,
    # still pending → stays. Registry source_uri points at the old Landing
    # path, so identification is by content hash (transit_hashes index).
    for aid, path in list(transit_hashes.items()):
        try:
            if aid in main_ids:
                _move_to_dir(path, transit_root, archive_root)
                stats["archived"] += 1
                transit_hashes.pop(aid, None)
            elif aid in rejected_artifacts or statuses_by_id.get(aid) == "no_text":
                _robust_delete(path)
                stats["deleted"] += 1
                transit_hashes.pop(aid, None)
        except Exception as exc:
            stats["errors"].append(f"{path.name}: {exc}")

    stats["pending_cleaned"] = cleanup_pending_delete(landing_root, transit_root)
    return stats
