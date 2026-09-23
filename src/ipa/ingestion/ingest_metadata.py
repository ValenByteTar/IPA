"""Señales derivadas por documento, escritas en el momento de la ingesta (Tier 0).

Lo que Tier 1 hoy re-deriva escaneando el corpus en cada ciclo idle, Tier 0 ya
lo conoce al ingerir: proveniencia (artifact bajo ``<landing>/web/**``), hash
normalizado, título, fecha de publicación y duplicados exactos contra main.
Todo va a tablas derivadas (``document_sources``, ``document_metadata``) —
``documents`` queda canónico.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ipa.ingestion.provenance import _domain_from_url, _source_url_from_text
from ipa.storage.document_store import DocumentStore


def _normalized_hash(text: str) -> str:
    # Fuente única: reporter_curation.normalized_hash — la comparación de
    # duplicados por URL/hash exige el mismo formato en ambos lados.
    from ipa.reporter.reporter_curation import normalized_hash
    return normalized_hash(text or "")


def _title_from_text(text: str) -> str:
    for line in (text or "").split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            # Los archivos scrapeados arrancan con la línea "Title: X" — el
            # prefijo no es parte del título.
            if stripped.lower().startswith("title:"):
                stripped = stripped[6:].strip()
            return stripped[:200]
    return ""


def _date_from_text(text: str) -> str:
    """Extrae la línea ``Date:`` que trafilatura embebe junto a ``Source:``."""
    for line in (text or "")[:2000].splitlines():
        line = line.strip()
        if line.lower().startswith("date:"):
            return line[5:].strip()
    return ""


def _load_scrape_report(landing_root: Path) -> dict[str, dict]:
    """{resolved saved_to path: record} desde scrape_report.json.

    El scraper lo escribe en su output dir (``Landing/web``); cuando el input
    de ingesta es el padre (``Landing``, default del CLI) se busca también en
    el subdir ``web/``.
    """
    root = Path(landing_root)
    report_path = root / "scrape_report.json"
    if not report_path.exists():
        report_path = root / "web" / "scrape_report.json"
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = data.get("results", [])
    out: dict[str, dict] = {}
    for rec in records:
        saved_to = rec.get("saved_to")
        if not saved_to:
            continue
        try:
            out[str(Path(saved_to).resolve())] = rec
        except (OSError, ValueError):
            out[saved_to] = rec
    return out


def _artifact_paths(landing_db_path: Path, artifact_ids: Iterable[str]) -> dict[str, str]:
    """{artifact_id: source_uri} desde la landing.db del corpus."""
    ldb = Path(landing_db_path)
    if not ldb.exists():
        return {}
    ids = list(artifact_ids)
    if not ids:
        return {}
    conn = sqlite3.connect(str(ldb))
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT artifact_id, source_uri FROM artifacts WHERE artifact_id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()
    return {row[0]: row[1] for row in rows if row[1]}


def record_ingest_metadata(
    store: DocumentStore,
    document_ids: Iterable[str],
    *,
    landing_db_path: str | Path | None = None,
    web_root: str | Path | None = None,
    scrape_report_dir: str | Path | None = None,
    main_store: DocumentStore | None = None,
) -> dict[str, int]:
    """Escribe metadata derivada para los docs recién ingeridos.

    - ``document_metadata``: normalized_hash / title / published_at / char_count
      para cualquier doc (cualquier proveniencia).
    - ``document_sources``: si el artifact vive bajo ``web_root`` →
      ``configured_scrape`` con url/quality/fecha del scrape_report o de las
      líneas ``Source:``/``Date:`` del propio texto (misma heurística que el
      backfill, pero en el momento — Tier 1 ya no necesita re-derivarla).
    - Dup exacto vs main: si ``main_store`` ya tiene el mismo normalized_hash →
      ``extra.duplicate_of_main`` (hecho; la decisión DUPLICATE la toma T1).
    """
    stats = {"meta": 0, "provenance": 0, "duplicates": 0}
    doc_ids = [d for d in document_ids if d]
    if not doc_ids:
        return stats

    # Qué ya está registrado — la ingesta es idempotente, no pisar provenance.
    try:
        recorded_sources = {
            r[0] for r in store._conn.execute(
                "SELECT document_id FROM document_sources").fetchall()
        }
    except sqlite3.Error:
        recorded_sources = set()
    # Solo las filas con hash completan la metadata: una fila creada por un
    # novelty_hint (sin normalized_hash) no debe impedir el relleno posterior.
    existing_meta = {
        r[0] for r in store._conn.execute(
            "SELECT document_id FROM document_metadata "
            "WHERE normalized_hash IS NOT NULL").fetchall()
    }

    artifact_map: dict[str, str] = {}
    doc_to_artifact: dict[str, str] = {}
    if web_root is not None and landing_db_path is not None:
        doc_to_artifact = {
            r[0]: r[1] for r in store._conn.execute(
                "SELECT document_id, artifact_id FROM documents "
                f"WHERE document_id IN ({','.join('?' * len(doc_ids))})",
                doc_ids).fetchall()
        }
        artifact_map = _artifact_paths(Path(landing_db_path), doc_to_artifact.values())

    web_root = Path(web_root).resolve() if web_root else None
    report_map = _load_scrape_report(Path(scrape_report_dir)) if scrape_report_dir else {}
    main_hashes: set[str] = set()
    if main_store is not None:
        try:
            main_hashes = {
                r[0] for r in main_store._conn.execute(
                    "SELECT normalized_hash FROM document_metadata "
                    "WHERE normalized_hash IS NOT NULL").fetchall()
            }
        except sqlite3.Error:
            main_hashes = set()

    for doc_id in doc_ids:
        has_meta = doc_id in existing_meta
        needs_source = web_root is not None and doc_id not in recorded_sources
        if has_meta and not needs_source:
            continue  # nada que escribir — evita el get_document

        # Resolver artifact/scrape_report antes de la metadata: la fecha del
        # reporte es más confiable que la línea Date: del texto.
        rec: dict = {}
        source_ok = False
        if needs_source:
            source_uri = artifact_map.get(doc_to_artifact.get(doc_id), "")
            if source_uri:
                try:
                    resolved = str(Path(source_uri).resolve())
                    rel = Path(resolved).relative_to(web_root)
                    # `web_root` puede ser <landing>/web (pipeline) o <landing>
                    # (CLI default) — solo los artifacts bajo web/** son
                    # configured_scrape; un archivo manual en Landing/ raíz no.
                    source_ok = web_root.name == "web" or bool(
                        rel.parts and rel.parts[0] == "web")
                    if source_ok:
                        rec = report_map.get(resolved, {})
                except (ValueError, OSError):
                    pass  # no es artifact del scraper configurado

        doc = store.get_document(doc_id)
        if doc is None:
            continue
        text = doc.text or ""

        # --- document_metadata ---
        extra: dict[str, Any] = {}
        if not has_meta:
            nhash = _normalized_hash(text)
            published = rec.get("date") or _date_from_text(text)
            if main_hashes and nhash in main_hashes:
                extra["duplicate_of_main"] = True
                stats["duplicates"] += 1
            store.put_doc_meta(
                doc_id,
                normalized_hash=nhash,
                title=_title_from_text(text),
                published_at=published or None,
                char_count=len(text),
                extra=extra or None,
            )
            stats["meta"] += 1

        # --- document_sources (solo si falta) ---
        if not source_ok:
            continue
        source_url = rec.get("url") or rec.get("canonical_url") or _source_url_from_text(text)
        published_at = rec.get("date") or _date_from_text(text) or None
        store.put_source(
            doc_id, source_url, _domain_from_url(source_url),
            "configured_scrape",
            float(rec.get("quality_score") or 0.0),
            published_at=published_at,
        )
        stats["provenance"] += 1

    store.commit()
    return stats


def backfill_doc_metadata(store: DocumentStore, *, limit: int | None = None) -> int:
    """Rellena document_metadata para docs legacy sin fila (una vez por corpus).

    Con ``limit`` se acota por llamada para no acaparar un ciclo idle; devuelve
    cuántas filas escribió.
    """
    # También cubre filas "hint-only": un novelty_hint escrito post-drain
    # crea la fila con normalized_hash NULL — no debe quedar sin metadata.
    rows = store._conn.execute(
        "SELECT d.document_id, d.text FROM documents d "
        "LEFT JOIN document_metadata m ON m.document_id = d.document_id "
        "WHERE d.tombstoned = 0 AND (m.document_id IS NULL "
        "OR m.normalized_hash IS NULL)"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    for doc_id, text in rows:
        text = text or ""
        store.put_doc_meta(
            doc_id,
            normalized_hash=_normalized_hash(text),
            title=_title_from_text(text),
            published_at=_date_from_text(text) or None,
            char_count=len(text),
        )
    if rows:
        store.commit()
    return len(rows)


def compute_novelty_hints(
    store: DocumentStore,
    lance: Any,
    main_store: DocumentStore,
    main_lance: Any,
) -> int:
    """Hint de novelty por doc (max coseno vs main + doc más cercano).

    Corre post-drain en Tier 0: los embeddings del corpus están recién
    calculados y los de main se cargan una sola vez. T1 usa el hint para
    saltear la carga completa de embeddings/textos históricos cuando el
    ``main_doc_count`` registrado sigue igual.
    """
    import numpy as np

    metas = store.all_doc_meta()
    doc_embs = lance.document_embeddings()
    universe = set(store.all_centroids()) or set(doc_embs)
    candidates = [
        doc_id for doc_id in universe
        if "novelty_hint" not in (metas.get(doc_id, {}).get("extra") or {})
    ]
    if not candidates:
        return 0
    main_embs = main_lance.document_embeddings()
    pairs = [(k, v) for k, v in main_embs.items() if v]
    if not pairs:
        return 0
    main_ids = [k for k, _ in pairs]
    mat = np.asarray([v for _, v in pairs], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    mat = mat / norms
    main_count = main_store.count_documents()
    # Token de versión del snapshot: MAX(stored_at) cambia solo al agregar
    # docs a main — permite a T1 refrescar el hint incrementalmente contra
    # los embeddings de los docs nuevos en vez de recargar todo el corpus.
    main_latest = (main_store._conn.execute(
        "SELECT MAX(stored_at) FROM documents WHERE tombstoned = 0"
    ).fetchone() or [None])[0]

    written = 0
    for doc_id in candidates:
        emb = doc_embs.get(doc_id)
        if not emb:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(vec)
        if n == 0:
            continue
        sims = mat @ (vec / n)
        idx = int(sims.argmax())
        store.put_doc_meta(doc_id, extra={
            "novelty_hint": {
                "max_cosine": float(sims[idx]),
                "nearest_doc_id": main_ids[idx],
                "main_doc_count": main_count,
                "main_latest_stored_at": main_latest,
            },
        })
        written += 1
    if written:
        store.commit()
    return written
