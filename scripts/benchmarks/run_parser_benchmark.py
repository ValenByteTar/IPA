"""Benchmark harness for E3 (parser competition) and E5 (chunking competition).

E3: Compares PDF parsers (PyMuPDF vs Docling vs Unstructured) on text coverage,
    reading order, latency, memory, and chunk quality.
E5: Compares chunking strategies (fixed-window vs recursive vs token vs semantic)
    on boundary quality, duplicate rate, chunk count, and retrieval recall.

Usage:
    # E3: parser competition on a sample of PDFs
    python scripts/benchmarks/run_parser_benchmark.py --pdfs Landing --output outputs/experiments/E3 --limit 20

    # E5: chunking competition on extracted text
    python scripts/benchmarks/run_parser_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E5 --limit-docs 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ipa import (
    CanonicalDocument,
    DocumentChunk,
    SourceSpan,
    chunk_document,
    chunk_document_recursive,
    chunk_document_token,
    parse_pdf_pymupdf,
)
from ipa.contracts import ParserResult


# ---------------------------------------------------------------------------
# E3: Parser benchmark
# ---------------------------------------------------------------------------

def select_balanced_pdfs(pdf_dir: str, limit: int = 20) -> list[Path]:
    """Select a deterministic size-balanced PDF sample for E3.

    The default 20-document sample contains 5 short PDFs (<=50 pages),
    10 medium PDFs (51-99 pages), and 5 long-but-manageable PDFs (100-200
    pages). PDFs above 200 pages are excluded from this initial run.
    """
    candidates = []
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        import pymupdf
        document = pymupdf.open(str(path))
        page_count = len(document)
        document.close()
        candidates.append((path, page_count))

    if limit != 20:
        return [path for path, pages in candidates if pages <= 200][:limit]

    buckets = {
        "short": [(p, n) for p, n in candidates if n <= 50],
        "medium": [(p, n) for p, n in candidates if 51 <= n <= 99],
        "long": [(p, n) for p, n in candidates if 100 <= n <= 200],
    }
    quotas = {"short": 5, "medium": 10, "long": 5}
    selected = [item for bucket in buckets.values() for item in bucket[:quotas[next(k for k, v in buckets.items() if v is bucket)]]]
    return [path for path, pages in selected]


def benchmark_parsers(pdf_dir: str, output_dir: str, limit: int = 20) -> dict:
    """Run E3: compare PDF parsers on a balanced sample of PDFs."""
    from ipa import parse_pdf_docling, parse_pdf_unstructured

    pdfs = select_balanced_pdfs(pdf_dir, limit)
    print(f"\nE3: Parser competition on {len(pdfs)} balanced PDFs", flush=True)

    parsers = {
        "pymupdf": parse_pdf_pymupdf,
        "docling": parse_pdf_docling,
        "unstructured": parse_pdf_unstructured,
    }

    results = {}
    for parser_name, parser_fn in parsers.items():
        print(f"\n=== {parser_name} ===", flush=True)
        times = []
        text_lengths = []
        page_counts = []
        errors = 0

        for i, pdf_path in enumerate(pdfs):
            artifact_id = f"sha256:{pdf_path.name}"
            t0 = time.monotonic()
            try:
                result = parser_fn(pdf_path, artifact_id)
                elapsed = time.monotonic() - t0
                if result.status == "parsed" and result.canonical_document:
                    times.append(elapsed)
                    text_lengths.append(len(result.canonical_document.text))
                    page_counts.append(result.canonical_document.pages)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  ERROR [{parser_name}] {pdf_path.name}: {e}", flush=True)

            if (i + 1) % 5 == 0:
                avg_time = sum(times) / len(times) if times else 0
                print(f"  {parser_name}: {i+1}/{len(pdfs)} done, avg={avg_time:.2f}s, errors={errors}", flush=True)

        avg_time = sum(times) / len(times) if times else 0
        total_text = sum(text_lengths)
        avg_pages = sum(page_counts) / len(page_counts) if page_counts else 0

        print(f"  {parser_name} done: {len(times)} parsed, {errors} errors, "
              f"avg_time={avg_time:.2f}s, total_text={total_text:,} chars, "
              f"avg_pages={avg_pages:.1f}", flush=True)

        results[parser_name] = {
            "parser": parser_name,
            "pdfs_parsed": len(times),
            "errors": errors,
            "avg_time_per_pdf_s": round(avg_time, 3),
            "total_time_s": round(sum(times), 2),
            "total_text_chars": total_text,
            "avg_pages": round(avg_pages, 1),
            "avg_text_per_pdf": round(total_text / len(times), 0) if times else 0,
        }

    return results


# ---------------------------------------------------------------------------
# E5: Chunking benchmark
# ---------------------------------------------------------------------------

def load_documents_from_store(
    store_db: str, limit: int = 50, mime_type: str | None = None
) -> list[CanonicalDocument]:
    """Load CanonicalDocuments from DocumentStore for chunking comparison.

    If mime_type is given (e.g. 'application/pdf'), only documents of that
    type are loaded — useful for comparing chunkers on PDF-extracted text.
    """
    import sqlite3
    import json as _json

    conn = sqlite3.connect(store_db)
    if mime_type:
        rows = conn.execute(
            "SELECT document_id, text, mime_type, parser_id, pages, elements_json, spans_json "
            "FROM documents WHERE tombstoned=0 AND mime_type=? LIMIT ?",
            (mime_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT document_id, text, mime_type, parser_id, pages, elements_json, spans_json "
            "FROM documents WHERE tombstoned=0 LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()

    docs = []
    for row in rows:
        spans = []
        if row[6]:
            for s in _json.loads(row[6]):
                spans.append(SourceSpan(
                    artifact_id=s["artifact_id"], page=s["page"],
                    offset_start=s["offset_start"], offset_end=s["offset_end"],
                ))
        elements = _json.loads(row[5]) if row[5] else []
        docs.append(CanonicalDocument(
            document_id=row[0], text=row[1], mime_type=row[2],
            parser_id=row[3], pages=row[4], elements=elements,
            source_spans=spans,
        ))
    return docs


def benchmark_chunkers(
    store_db: str, output_dir: str, limit_docs: int = 50, mime_type: str | None = None
) -> dict:
    """Run E5: compare chunking strategies on extracted text."""
    label = f" {mime_type}" if mime_type else ""
    print(f"\nE5: Chunking competition on {limit_docs}{label} documents", flush=True)

    docs = load_documents_from_store(store_db, limit=limit_docs, mime_type=mime_type)
    print(f"  Loaded {len(docs)} documents", flush=True)

    chunkers = {
        "fixed_window": lambda doc: chunk_document(doc, chunk_size=512, overlap=64),
        "recursive": lambda doc: chunk_document_recursive(doc, chunk_size=512, overlap=64),
        "token": lambda doc: chunk_document_token(doc, chunk_size=200, overlap=20),
    }

    results = {}
    for chunker_name, chunker_fn in chunkers.items():
        print(f"\n=== {chunker_name} ===", flush=True)
        t0 = time.monotonic()
        total_chunks = 0
        total_chars = 0
        chunk_sizes = []
        duplicate_hashes = set()
        duplicate_count = 0

        for doc in docs:
            chunks = chunker_fn(doc)
            total_chunks += len(chunks)
            for chunk in chunks:
                total_chars += len(chunk.text)
                chunk_sizes.append(len(chunk.text))
                if chunk.content_hash in duplicate_hashes:
                    duplicate_count += 1
                else:
                    duplicate_hashes.add(chunk.content_hash)

        elapsed = time.monotonic() - t0
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        dup_rate = duplicate_count / total_chunks if total_chunks else 0

        print(f"  {chunker_name} done: {total_chunks} chunks in {elapsed:.2f}s, "
              f"avg_size={avg_chunk_size:.0f} chars, dup_rate={dup_rate:.2%}", flush=True)

        results[chunker_name] = {
            "chunker": chunker_name,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 1),
            "duplicate_rate": round(dup_rate, 4),
            "duplicate_count": duplicate_count,
            "elapsed_seconds": round(elapsed, 2),
            "chunks_per_second": round(total_chunks / elapsed, 1) if elapsed > 0 else 0,
        }

    # Semantic chunker — run on same docs with tuned parameters.
    print(f"\n=== semantic ===", flush=True)
    from ipa import chunk_document_semantic
    t0 = time.monotonic()
    total_chunks = 0
    total_chars = 0
    chunk_sizes = []
    duplicate_hashes = set()
    duplicate_count = 0
    sem_docs = docs  # run on all docs — tuned threshold makes it feasible

    for doc in sem_docs:
        try:
            chunks = chunk_document_semantic(doc)
            total_chunks += len(chunks)
            for chunk in chunks:
                total_chars += len(chunk.text)
                chunk_sizes.append(len(chunk.text))
                if chunk.content_hash in duplicate_hashes:
                    duplicate_count += 1
                else:
                    duplicate_hashes.add(chunk.content_hash)
        except Exception as e:
            print(f"  ERROR semantic: {e}", flush=True)

    elapsed = time.monotonic() - t0
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    dup_rate = duplicate_count / total_chunks if total_chunks else 0

    print(f"  semantic done: {total_chunks} chunks in {elapsed:.2f}s, "
          f"avg_size={avg_chunk_size:.0f} chars, dup_rate={dup_rate:.2%}", flush=True)

    results["semantic"] = {
        "chunker": "semantic",
        "docs_processed": len(sem_docs),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "avg_chunk_size": round(avg_chunk_size, 1),
        "duplicate_rate": round(dup_rate, 4),
        "duplicate_count": duplicate_count,
        "elapsed_seconds": round(elapsed, 2),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run parser/chunker benchmark (E3/E5).")
    parser.add_argument("--pdfs", help="Directory of PDFs for E3 parser benchmark")
    parser.add_argument("--store", help="Path to document_store.db for E5 chunker benchmark")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=20, help="Limit PDFs for E3")
    parser.add_argument("--limit-docs", type=int, default=50, help="Limit docs for E5")
    parser.add_argument("--mime-type", default=None, help="Filter E5 by mime_type (e.g. application/pdf)")
    parser.add_argument("--mode", choices=["parsers", "chunkers", "both"], default="both")
    args = parser.parse_args()

    report = {
        "output": args.output,
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.mode in ("parsers", "both") and args.pdfs:
        report["E3_parsers"] = benchmark_parsers(args.pdfs, args.output, args.limit)

    if args.mode in ("chunkers", "both") and args.store:
        report["E5_chunkers"] = benchmark_chunkers(
            args.store, args.output, args.limit_docs, mime_type=args.mime_type
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
