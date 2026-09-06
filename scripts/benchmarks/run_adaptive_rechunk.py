"""Analyze/apply adaptive rechunking without modifying the source store."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan
from ipa.ingestion.adaptive_chunker import adaptive_rechunk, lexical_density


def span_from_json(raw: str | None) -> SourceSpan | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not data:
        return None
    return SourceSpan(**data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.store)
    docs = conn.execute(
        "SELECT document_id, artifact_id, parser_id, mime_type, pages, text, elements_json, spans_json "
        "FROM documents WHERE tombstoned=0 ORDER BY document_id"
    ).fetchall()
    chunks_by_doc: dict[str, list[DocumentChunk]] = {}
    for row in conn.execute(
        "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
        "FROM chunks WHERE tombstoned=0 ORDER BY document_id, rowid"
    ):
        chunks_by_doc.setdefault(row[1], []).append(DocumentChunk(
            chunk_id=row[0], document_id=row[1], content_hash=row[2], text=row[3],
            metadata=json.loads(row[4] or "{}"), source_span=span_from_json(row[5]),
        ))
    conn.close()

    results = []
    activated = 0
    for row in docs:
        doc_id, artifact_id, parser_id, mime_type, pages, text, elements_json, spans_json = row
        doc = CanonicalDocument(
            document_id=doc_id, pages=pages, text=text, parser_id=parser_id,
            mime_type=mime_type, elements=json.loads(elements_json or "[]"),
            source_spans=[SourceSpan(**s) for s in json.loads(spans_json or "[]")],
        )
        result = adaptive_rechunk(doc, chunks_by_doc.get(doc_id, []))
        if result.reason != "no_action":
            activated += 1
        results.append({
            "document_id": result.document_id,
            "original_chunk_count": result.original_chunk_count,
            "final_chunk_count": result.final_chunk_count,
            "merged_count": result.merged_count,
            "fallback_rechunk_count": result.fallback_rechunk_count,
            "output_delta": result.final_chunk_count - result.original_chunk_count,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content_hash": c.content_hash,
                    "text": c.text,
                    "metadata": c.metadata,
                    "source_span": c.source_span.__dict__ if c.source_span else None,
                }
                for c in result.chunks
            ],
            "reason": result.reason,
            "original_low_density_chunks": sum(
                lexical_density(c.text) < 0.4 for c in chunks_by_doc.get(doc_id, [])
            ),
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "store": args.store,
        "documents": len(results),
        "activated_documents": activated,
        "total_original_chunks": sum(r["original_chunk_count"] for r in results),
        "total_final_chunks": sum(r["final_chunk_count"] for r in results),
        "total_merged": sum(r["merged_count"] for r in results),
        "total_fallback_rechunk": sum(r["fallback_rechunk_count"] for r in results),
        "output_delta": sum(r["output_delta"] for r in results),
        "results": results,
    }
    path = output / "adaptive_rechunk_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Documents analyzed: {len(results):,}")
    print(f"Documents activated: {activated:,}")
    print(f"Original chunks: {report['total_original_chunks']:,}")
    print(f"Final chunks: {report['total_final_chunks']:,}")
    print(f"Merged: {report['total_merged']:,}")
    print(f"Fallback rechunk count: {report['total_fallback_rechunk']:,}")
    print(f"Output delta: {report['output_delta']:+,}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
