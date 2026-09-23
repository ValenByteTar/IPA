"""Auditoría de salud del corpus — detección automática, reparación gated.

Dos capas con costos distintos:

- **Lógica** (barata, cooldown corto): consume las señales Tier 0 ya
  persistidas (``document_metadata``/``document_sources``) — nada de
  escanear textos. Detecta docs vacíos vivos, flags ``duplicate_of_main``
  fugados, ``normalized_hash`` repetido entre vivos, hints de novelty
  stale, cola de backfill de metadata y gaps de proveniencia.
- **Física** (cara, cooldown largo, siempre forzada): compara los índices
  reales (store ↔ BM25 meta ↔ FTS ↔ LanceDB) por *sets de chunk_id* —
  nunca por texto, porque BM25/LanceDB pueden guardar la representación
  enriquecida (``enriched_text()``) mientras ``chunks.text`` queda
  canónico. Detecta drift por kills/locks que las señales lógicas no ven:
  chunks faltantes/huérfanos, ``chunk_id`` duplicados en LanceDB y spam
  chunks (mismo ``content_hash`` en ≥3 docs o ≥2 dominios).

Salida: ``outputs/agent/index_health.json`` (sección por capa con su propio
``checked_at`` — capas independientes que se mergean read-modify-write).

Política: el audit es **read-only**. Escribe solo el JSON de salud; las
acciones (tombstone, dedupe, restore) quedan como herramientas ops con
dry-run — ver ``docs/plans/tier0-signals-idle-optimization.md``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs" / "agent" / "index_health.json"

# Umbrales revisables (env-tunable): el audit reporta, no actúa.
SPAM_MIN_DOCS = int(os.environ.get("IPA_AUDIT_SPAM_MIN_DOCS", "3") or 3)
DUP_SAMPLE = int(os.environ.get("IPA_AUDIT_DUP_SAMPLE", "10") or 10)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Capa lógica (señales Tier 0) ──────────────────────────────────────────

def run_logical_checks(
    conn: sqlite3.Connection,
    *,
    cluster_conn: sqlite3.Connection | None = None,
    is_main: bool = True,
) -> dict[str, Any]:
    """Checks baratos sobre tablas derivadas — sin escanear textos.

    ``is_main`` desactiva los checks que solo tienen sentido contra el main
    (un doc del staging flaggeado ``duplicate_of_main`` es trabajo de T1,
    no una anomalía).
    """
    out: dict[str, Any] = {}

    live = {r[0] for r in conn.execute(
        "SELECT document_id FROM documents WHERE tombstoned = 0")}

    # Meta: coverage de la tabla derivada + señales persistidas.
    meta: dict[str, dict] = {}
    try:
        for did, nhash, cc, ej in conn.execute(
                "SELECT document_id, normalized_hash, char_count, extra_json "
                "FROM document_metadata"):
            meta[did] = {"normalized_hash": nhash, "char_count": cc,
                         "extra": ej}
    except sqlite3.Error:
        out["meta_table"] = "missing"
        return out

    live_meta = {d: m for d, m in meta.items() if d in live}
    out["missing_meta"] = sorted(live - set(meta))[:DUP_SAMPLE]
    out["missing_meta_count"] = len(live - set(meta))
    out["null_hash_count"] = sum(
        1 for m in live_meta.values() if not m["normalized_hash"])

    # char_count NULL = no computado (fila creada solo por un marker como
    # dedupe_url/novelty_hint) — no es un doc vacío. Solo cuenta el 0 real.
    out["empty_docs"] = sorted(
        d for d, m in live_meta.items() if m["char_count"] == 0)
    out["empty_docs_count"] = len(out["empty_docs"])

    # normalized_hash repetido entre docs vivos → dup real (mismo formato
    # sha256: que reporter_curation — Tier 0 lo persiste al ingerir).
    dup_hashes = conn.execute(
        "SELECT m.normalized_hash, COUNT(*) c FROM document_metadata m "
        "JOIN documents d ON d.document_id = m.document_id AND d.tombstoned = 0 "
        "WHERE m.normalized_hash IS NOT NULL "
        "GROUP BY m.normalized_hash HAVING c > 1").fetchall()
    out["dup_hashes"] = [
        {"hash": h, "count": c,
         "doc_ids": [r[0] for r in conn.execute(
             "SELECT m.document_id FROM document_metadata m "
             "JOIN documents d ON d.document_id = m.document_id "
             "AND d.tombstoned = 0 WHERE m.normalized_hash = ?",
             (h,)).fetchall()]}
        for h, c in dup_hashes[:DUP_SAMPLE]]
    out["dup_hashes_count"] = len(dup_hashes)

    # duplicate_of_main flag ∧ doc sigue vivo ∧ sin decisión DUPLICATE →
    # el gate Tier0→T1 no actuó (fuga). En main el flag no debería existir
    # (main es la referencia); si aparece es anomalía igual.
    dup_flagged = [
        d for d, m in live_meta.items()
        if m["extra"] and "duplicate_of_main" in (json.loads(m["extra"]) or {})
    ]
    if dup_flagged and cluster_conn is not None:
        try:
            dup_decided = {r[0] for r in cluster_conn.execute(
                "SELECT document_id FROM curation_decisions "
                "WHERE json_extract(payload_json, '$.decision') = 'duplicate'")}
        except sqlite3.Error:
            dup_decided = set()
        out["dup_flag_leaks"] = sorted(d for d in dup_flagged
                                       if d not in dup_decided)
    else:
        out["dup_flag_leaks"] = sorted(dup_flagged)
    out["dup_flag_leaks_count"] = len(out["dup_flag_leaks"])

    # Hints de novelty stale: main_doc_count del hint ≠ count actual de
    # main (cualquier promoción los invalida — visibilidad del trade-off).
    if is_main:
        out["stale_hints_count"] = 0  # hints de main vs main no aplican
    else:
        main_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE tombstoned = 0").fetchone()[0]
        out["stale_hints_count"] = sum(
            1 for m in live_meta.values()
            if m["extra"]
            and (h := (json.loads(m["extra"]) or {}).get("novelty_hint"))
            and h.get("main_doc_count") != main_count)

    # Proveniencia configured_scrape sin URL → el scrape_report no tenía
    # la entrada (gap de registro, no de contenido).
    try:
        out["scrape_no_url"] = sorted(r[0] for r in conn.execute(
            "SELECT s.document_id FROM document_sources s "
            "JOIN documents d ON d.document_id = s.document_id "
            "AND d.tombstoned = 0 "
            "WHERE s.provenance = 'configured_scrape' "
            "AND (s.source_url IS NULL OR s.source_url = '')"))
        out["scrape_no_url_count"] = len(out["scrape_no_url"])
    except sqlite3.Error:
        out["scrape_no_url_count"] = 0

    return out


# ── Capa física (índices reales — drift por kills/locks) ──────────────────

def run_physical_checks(
    conn: sqlite3.Connection,
    corpus: Path,
    *,
    lance_table: Any = None,
) -> dict[str, Any]:
    """Comparación por chunk_id entre store, BM25 meta, FTS y LanceDB.

    Nunca compara texto: ``enriched_text()`` hace que BM25/LanceDB guarden
    la representación derivada mientras ``chunks.text`` queda canónico.
    ``lance_table`` se inyecta en tests (la tabla real se abre lazy).
    """
    out: dict[str, Any] = {}
    live_chunks = {r[0] for r in conn.execute(
        "SELECT chunk_id FROM chunks WHERE tombstoned = 0")}

    bm25_db = Path(corpus) / "bm25_index.db"
    if bm25_db.exists():
        b = sqlite3.connect(f"file:{bm25_db}?mode=ro", uri=True, timeout=30)
        try:
            meta_ids = {r[0] for r in b.execute(
                "SELECT chunk_id FROM chunks_meta WHERE tombstoned = 0")}
            fts_ids = {r[0] for r in b.execute(
                "SELECT chunk_id FROM chunks_fts")}
        finally:
            b.close()
        out["bm25"] = {
            "meta": len(meta_ids), "fts": len(fts_ids),
            "meta_missing": sorted(live_chunks - meta_ids)[:DUP_SAMPLE],
            "meta_orphans": sorted(meta_ids - live_chunks)[:DUP_SAMPLE],
            "fts_missing": sorted(live_chunks - fts_ids)[:DUP_SAMPLE],
            "fts_orphans": sorted(fts_ids - live_chunks)[:DUP_SAMPLE],
            "meta_missing_count": len(live_chunks - meta_ids),
            "meta_orphans_count": len(meta_ids - live_chunks),
            "fts_missing_count": len(live_chunks - fts_ids),
            "fts_orphans_count": len(fts_ids - live_chunks),
        }
    else:
        out["bm25"] = None

    # LanceDB: chunk_ids + duplicados (una inserción parcial puede dejar
    # el mismo chunk_id dos veces si el add_chunks no borró antes).
    lance_ids: set[str] = set()
    lance_dups = 0
    tbl = lance_table
    if tbl is None:
        lance_dir = Path(corpus) / "vector" / "lancedb"
        if lance_dir.exists():
            try:
                from ipa.indexes.lancedb_index import LanceDBIndex
                lx = LanceDBIndex(lance_dir, vector_dim=1024)
                tbl = lx._table if lx.is_queryable() else None
            except Exception:
                tbl = None
    if tbl is not None:
        # Proyección de columna (sin vectores); fallback to_arrow() para
        # fakes/tablas sin query API. None = tabla ilegible.
        from ipa.indexes.lancedb_index import table_chunk_id_list
        ids = table_chunk_id_list(tbl)
        if ids is None:
            out["lance_error"] = "no se pudo leer chunk_ids de LanceDB"
        else:
            lance_ids = set(ids)
            lance_dups = len(ids) - len(lance_ids)
    out["lance"] = {
        "vectors": len(lance_ids),
        "duplicate_chunk_ids": lance_dups,
        "missing_count": len(live_chunks - lance_ids),
        "orphans_count": len(lance_ids - live_chunks),
        "missing": sorted(live_chunks - lance_ids)[:DUP_SAMPLE],
        "orphans": sorted(lance_ids - live_chunks)[:DUP_SAMPLE],
    }

    # Spam chunks: mismo content_hash en ≥3 docs vivos distintos (chrome
    # site-wide tipo nav de arXiv) o en ≥2 dominios (sindicación real —
    # NO se tombstonea automático: puede ser contenido legítimo).
    rows = conn.execute(
        "SELECT c.content_hash, COUNT(DISTINCT c.document_id) nd, "
        "COUNT(DISTINCT s.source_domain) ndom "
        "FROM chunks c "
        "JOIN documents d ON d.document_id = c.document_id AND d.tombstoned = 0 "
        "LEFT JOIN document_sources s ON s.document_id = c.document_id "
        "WHERE c.tombstoned = 0 "
        "GROUP BY c.content_hash HAVING nd >= 2").fetchall()
    spam_site = [h for h, nd, _ in rows if nd >= SPAM_MIN_DOCS]
    spam_cross = [h for h, nd, ndom in rows if nd >= 2 and (ndom or 0) >= 2]
    out["spam_chunks"] = {
        "same_site_hashes": len(spam_site),
        "cross_domain_hashes": len(spam_cross),
        "same_site_sample": spam_site[:DUP_SAMPLE],
        "cross_domain_sample": spam_cross[:DUP_SAMPLE],
    }
    out["chunks_live"] = len(live_chunks)
    return out


# ── Entrada + persistencia ────────────────────────────────────────────────

def _status(checks: dict[str, Any]) -> str:
    """ok / warn / fail según severidad de los hallazgos."""
    fail_keys = ("meta_missing_count", "fts_missing_count",
                 "missing_count", "duplicate_chunk_ids")
    for section in checks.values():
        if not isinstance(section, dict):
            continue
        for k, v in section.items():
            if k in fail_keys and isinstance(v, int) and v > 0:
                return "fail"
            if k in ("bm25", "lance") and isinstance(v, dict):
                for kk, vv in v.items():
                    if kk in fail_keys and isinstance(vv, int) and vv > 0:
                        return "fail"
    warn_keys = ("empty_docs_count", "dup_hashes_count", "dup_flag_leaks_count",
                 "missing_meta_count", "null_hash_count", "stale_hints_count",
                 "scrape_no_url_count", "same_site_hashes",
                 "cross_domain_hashes", "meta_orphans_count",
                 "fts_orphans_count", "orphans_count")
    for section in checks.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if k in warn_keys and isinstance(v, int) and v > 0:
                    return "warn"
            if k in ("bm25", "lance", "spam_chunks") and isinstance(v, dict):
                for kk, vv in v.items():
                    if kk in warn_keys and isinstance(vv, int) and vv > 0:
                        return "warn"
    return "ok"


def run_index_audit(
    corpus: Path | str,
    *,
    layer: str = "both",          # "logical" | "physical" | "both"
    cluster_db: Path | str | None = None,
    output: Path | str | None = None,
    is_main: bool = True,
) -> dict[str, Any]:
    """Corre las capas pedidas y mergea en ``index_health.json``.

    Read-only sobre el corpus: solo abre conexiones RO y escribe el JSON.
    Devuelve el payload completo (capas viejas preservadas si solo corrió
    una).
    """
    corpus = Path(corpus)
    store_db = corpus / "document_store.db"
    out_path = Path(output) if output else DEFAULT_OUTPUT

    previous: dict[str, Any] = {}
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}

    payload: dict[str, Any] = {
        "corpus": str(corpus),
        "corpus_name": corpus.name,
        "checked_at": _now(),
        "logical": previous.get("logical"),
        "physical": previous.get("physical"),
    }

    conn = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True, timeout=30)
    try:
        cluster_conn = None
        cdb = Path(cluster_db) if cluster_db else (
            ROOT / "outputs" / "agent" / "topic_clusters.db")
        if cdb.exists():
            cluster_conn = sqlite3.connect(
                f"file:{cdb}?mode=ro", uri=True, timeout=30)
        try:
            if layer in ("logical", "both"):
                payload["logical"] = {
                    "checked_at": _now(),
                    **run_logical_checks(
                        conn, cluster_conn=cluster_conn, is_main=is_main),
                }
            if layer in ("physical", "both"):
                payload["physical"] = {
                    "checked_at": _now(),
                    **run_physical_checks(conn, corpus),
                }
        finally:
            if cluster_conn is not None:
                cluster_conn.close()
    finally:
        conn.close()

    payload["status"] = _status(
        {k: v for k, v in payload.items()
         if k in ("logical", "physical") and isinstance(v, dict)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(out_path)
    return payload


__all__ = [
    "DEFAULT_OUTPUT", "run_logical_checks", "run_physical_checks",
    "run_index_audit",
]
