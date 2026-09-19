"""Enriquecimiento de chunks por LLM — resumen + queries sintéticas + re-embed.

Core importable del ex-worker `scripts/operations/workers/run_enrichment_exl3.py`
(cadena del Orchestrator, deprecada). La diferencia de diseño: este módulo NO
carga un modelo — corre sobre el provider ya cargado del pase Tier 2 profundo
(ExL3 9B batched, u Ollama en serial vía `generate_many`). En 6 GB de VRAM no
caben dos modelos: la tarea reusa el motor del pase.

Por chunk genera:
  1. Summary: 1-2 oraciones, prepend como "[Summary] ..."
  2. Queries sintéticas: 3 preguntas, prepend como "[Questions] ..."

Después re-embede los chunks enriquecidos en LanceDB (híbrido denso+sparse)
para que el retrieval vectorial se beneficie del texto enriquecido.

Resumable y con checkpoint durable:
  - chunks ya enriquecidos (text con prefijo "[Summary]") se saltean;
  - enrichment.embedding_status: pending → complete solo cuando LanceDB
    aceptó el batch (un corte a mitad deja un checkpoint recuperable);
  - metadata.enrichment.canonical_text preserva el texto original.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ipa.agentic.batch_llm import generate_many
from ipa.contracts import DocumentChunk
from ipa.ingestion.continuous_pipeline import should_summarize

MAX_NEW_TOKENS = 150  # summary + 3 questions
TEMPERATURE = 0.3
TEXT_LIMIT = 2000     # chars del chunk que entran al prompt
REEMBED_BATCH = 192   # batch de BGE-M3 para re-embedding

ENRICHMENT_VERSION = "exl3-v2-9b"

ENRICH_SYSTEM = (
    "You are a document enrichment system. "
    "First, summarize the text in 1-2 sentences. "
    "Then, generate 3 questions that this text would answer. "
    "Format your response as:\n"
    "SUMMARY: <your summary>\n"
    "Q1: <question 1>\n"
    "Q2: <question 2>\n"
    "Q3: <question 3>"
)

ENRICH_PROMPT = (
    'Enrich this text with a summary and 3 synthetic questions.\n\n'
    'Text:\n'
    '"""\n'
    '{text}\n'
    '"""\n'
)

_THINK_END = chr(60) + "/think" + chr(62)


def enrichment_messages(text: str) -> list[dict[str, str]]:
    """Mensajes chat para el provider (no_think del provider fuerza directa)."""
    return [
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user", "content": ENRICH_PROMPT.format(text=text[:TEXT_LIMIT])},
    ]


def parse_enrichment(raw_output: str) -> tuple[str, list[str]]:
    """Parsea la salida del LLM a (summary, questions).

    Formato esperado:
      SUMMARY: <summary>
      Q1: <q1>
      Q2: <q2>
      Q3: <q3>
    """
    if _THINK_END in raw_output:
        raw_output = raw_output.split(_THINK_END, 1)[1].strip()

    summary = ""
    questions: list[str] = []
    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.upper().startswith("Q") and ":" in line:
            q = line.split(":", 1)[1].strip()
            if len(q) > 5:
                questions.append(q)

    if not summary and not questions:
        summary = raw_output.strip()[:200]
    return summary, questions


def build_enriched_text(summary: str, questions: list[str], original: str) -> str:
    """Texto enriquecido: [Summary] + [Questions] prepend al original."""
    parts = []
    if summary:
        parts.append(f"[Summary] {summary}")
    if questions:
        q_block = "\n".join(f"Q: {q}" for q in questions)
        parts.append(f"[Questions]\n{q_block}")
    parts.append(original)
    return "\n\n".join(parts)


def _metadata(text: str | None) -> dict:
    try:
        value = json.loads(text or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _enriched_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ChunkScan:
    """Resultado del escaneo de la tabla chunks."""

    to_process: list[tuple[str, str, str, str]] = field(default_factory=list)
    pending_reembed: list[tuple[str, str, str, str]] = field(default_factory=list)
    total: int = 0
    already_enriched: int = 0
    skipped: int = 0

    @property
    def pending(self) -> int:
        return len(self.to_process) + len(self.pending_reembed)


def scan_chunks(conn: sqlite3.Connection, *, min_chars: int = 800,
                limit: int | None = None) -> ChunkScan:
    """Clasifica la tabla chunks: a procesar / re-embed pendiente / listos.

    limit acota to_process (el pase Tier 2 es incremental — el resto queda
    para el próximo idle profundo). pending_reembed nunca se acota: es
    trabajo ya generado que solo falta re-embedir.
    """
    scan = ChunkScan()
    rows = conn.execute(
        "SELECT chunk_id, document_id, text, content_hash, metadata_json "
        "FROM chunks"
    ).fetchall()
    scan.total = len(rows)
    for chunk_id, doc_id, text, content_hash, metadata_json in rows:
        meta = _metadata(metadata_json)
        if text.startswith("[Summary]"):
            scan.already_enriched += 1
            # Re-embed pendiente: checkpoint no commiteado o texto cambió.
            enr = meta.get("enrichment", {})
            if (enr.get("embedding_status") != "complete"
                    or enr.get("text_hash") != _enriched_hash(text)):
                scan.pending_reembed.append((chunk_id, doc_id, text, content_hash))
            continue
        if limit is not None and len(scan.to_process) >= limit:
            continue
        chunk = DocumentChunk(
            chunk_id=chunk_id, document_id=doc_id,
            text=text, content_hash=content_hash,
        )
        if should_summarize(chunk, min_chars=min_chars):
            scan.to_process.append((chunk_id, doc_id, text, content_hash))
        else:
            scan.skipped += 1
    return scan


def count_pending(store_db: Path, *, min_chars: int = 800) -> int:
    """Cuántos chunks necesitan trabajo (scan barato, sin cargar nada)."""
    conn = sqlite3.connect(str(store_db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return scan_chunks(conn, min_chars=min_chars).pending
    finally:
        conn.close()


def write_enrichment(conn: sqlite3.Connection, chunk_id: str,
                     enriched_text: str, original_text: str) -> None:
    """Marca el chunk como enriquecido (embedding_status=pending)."""
    row = conn.execute(
        "SELECT metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    meta = _metadata(row[0] if row else None)
    meta.setdefault("enrichment", {}).update({
        "version": ENRICHMENT_VERSION,
        "status": "enriched",
        "embedding_status": "pending",
        "text_hash": _enriched_hash(enriched_text),
        "enriched_at": time.time(),
    })
    meta["enrichment"].setdefault("canonical_text", original_text)
    conn.execute(
        "UPDATE chunks SET text = ?, metadata_json = ? WHERE chunk_id = ?",
        (enriched_text, json.dumps(meta, ensure_ascii=False), chunk_id),
    )


def reembed_batch(lancedb: Any, embedding: Any,
                  chunks: list[tuple[str, str, str, str]]) -> int:
    """Re-embede chunks enriquecidos en LanceDB (denso+sparse en un paso).

    add_chunks es idempotente (borra chunk_ids existentes antes de insertar).
    """
    if not chunks:
        return 0
    objs = [
        DocumentChunk(chunk_id=c[0], document_id=c[1], text=c[2], content_hash=c[3])
        for c in chunks
    ]
    dense, sparse = embedding.embed_texts_hybrid([c.text for c in objs])
    lancedb.add_chunks(objs, dense, sparse_weights=sparse)
    return len(chunks)


def mark_reembedded(conn: sqlite3.Connection,
                    chunks: Iterable[tuple[str, str, str, str]]) -> None:
    """embedding_status=complete SOLO después de que LanceDB aceptó el batch."""
    for chunk_id, _, text, _ in chunks:
        row = conn.execute(
            "SELECT metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        meta = _metadata(row[0] if row else None)
        meta.setdefault("enrichment", {}).update({
            "embedding_status": "complete",
            "embedded_at": time.time(),
            "text_hash": _enriched_hash(text),
        })
        conn.execute(
            "UPDATE chunks SET metadata_json = ? WHERE chunk_id = ?",
            (json.dumps(meta, ensure_ascii=False), chunk_id),
        )
    conn.commit()


def _flush_reembed(conn: sqlite3.Connection, lancedb: Any, embedding: Any,
                   pending: list[tuple[str, str, str, str]],
                   reembed_batch_size: int) -> int:
    """Drena la cola de re-embed en lotes; devuelve cuántos se re-embedieron."""
    done = 0
    while pending:
        batch, pending[:] = pending[:reembed_batch_size], pending[reembed_batch_size:]
        conn.commit()  # DB consistente antes de tocar LanceDB
        done += reembed_batch(lancedb, embedding, batch)
        mark_reembedded(conn, batch)
    return done


def enrich_chunks(
    provider: Any,
    store_db: str | Path,
    *,
    lancedb: Any = None,
    embedding: Any = None,
    limit: int | None = None,
    min_chars: int = 800,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    reembed_batch_size: int = REEMBED_BATCH,
    should_abort: Callable[[], bool] | None = None,
    log: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Pase de enriquecimiento sobre el provider ya cargado.

    provider: ExL3 (generate_chat_batch → lotes de provider.batch_size) u
        Ollama (generate_chat serial — generate_many lo resuelve).
    lancedb/embedding: si alguno es None, los chunks quedan enriquecidos con
        embedding_status=pending y se re-embeden en un pase posterior.
    should_abort: se consulta entre lotes (preempción del idle scheduler).
    Devuelve stats del pase.
    """
    log = log or (lambda *a, **k: None)
    abort = should_abort or (lambda: False)
    stats: dict[str, Any] = {
        "enriched": 0, "reembedded": 0, "errors": 0, "aborted": False,
    }

    conn = sqlite3.connect(str(store_db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        scan = scan_chunks(conn, min_chars=min_chars, limit=limit)
        stats.update({
            "total": scan.total, "already_enriched": scan.already_enriched,
            "skipped": scan.skipped, "pending": scan.pending,
        })
        if not scan.pending:
            return stats

        can_embed = lancedb is not None and embedding is not None
        pending_reembed = list(scan.pending_reembed)

        # Recuperar re-embeds de corridas interrumpidas (trabajo ya generado).
        if pending_reembed and can_embed:
            stats["reembedded"] += _flush_reembed(
                conn, lancedb, embedding, pending_reembed, reembed_batch_size)

        batch_size = max(1, int(getattr(provider, "batch_size", 1) or 1))
        for start in range(0, len(scan.to_process), batch_size):
            if abort():
                stats["aborted"] = True
                break
            batch = scan.to_process[start:start + batch_size]
            convs = [enrichment_messages(text) for _, _, text, _ in batch]
            results = generate_many(
                provider, convs,
                max_new_tokens=max_new_tokens, temperature=temperature)

            for (chunk_id, doc_id, text, content_hash), (raw, err) in zip(batch, results):
                if err or len((raw or "").strip()) < 10:
                    stats["errors"] += 1
                    continue
                summary, questions = parse_enrichment(raw)
                if not summary and not questions:
                    stats["errors"] += 1
                    continue
                enriched_text = build_enriched_text(summary, questions, text)
                write_enrichment(conn, chunk_id, enriched_text, text)
                stats["enriched"] += 1
                pending_reembed.append((chunk_id, doc_id, enriched_text, content_hash))

            conn.commit()  # checkpoint durable por lote

            if can_embed and len(pending_reembed) >= reembed_batch_size:
                stats["reembedded"] += _flush_reembed(
                    conn, lancedb, embedding, pending_reembed, reembed_batch_size)

            done = min(start + batch_size, len(scan.to_process))
            log(f"  [enrich] {done}/{len(scan.to_process)} "
                f"enriched={stats['enriched']} reembedded={stats['reembedded']} "
                f"errors={stats['errors']}")

        # Flush final de lo que quedó bajo el umbral.
        conn.commit()
        if pending_reembed and can_embed:
            stats["reembedded"] += _flush_reembed(
                conn, lancedb, embedding, pending_reembed, reembed_batch_size)
        stats["pending_reembed"] = len(pending_reembed)
        return stats
    finally:
        conn.close()


__all__ = [
    "ENRICH_SYSTEM", "ENRICH_PROMPT", "ENRICHMENT_VERSION",
    "MAX_NEW_TOKENS", "TEMPERATURE", "REEMBED_BATCH",
    "ChunkScan", "scan_chunks", "count_pending",
    "enrichment_messages", "parse_enrichment", "build_enriched_text",
    "write_enrichment", "reembed_batch", "mark_reembedded",
    "enrich_chunks",
]
