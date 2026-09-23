"""Fast path runner â€” orchestrates the full ingestion pipeline.

Pipeline:
    artifact â†’ Landing Zone â†’ MIME routing â†’ parser â†’ CanonicalDocument
    â†’ chunker â†’ DocumentStore â†’ Index â†’ first_queryable

The index backend is configurable:
    - 'bm25' (default, Stage 1): SQLite FTS5 â€” single-file, zero-dep
    - 'tantivy' (E6 winner): Rust Tantivy â€” faster, better ranking

This path does NOT wait for embeddings, LLM extraction, or enrichment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ipa.indexes.bm25_index import BM25Index
from ipa.ingestion.chunker import chunk_document
from ipa.ingestion.content_safety import (
    QuarantineConfig, QuarantineError, quarantine_file,
    detect_file_type_from_path,
)
from ipa.storage.document_store import DocumentStore
from ipa.ingestion.landing_zone import LandingZone
from ipa.ingestion.mime_router import detect_mime, route_to_parser
from ipa.ingestion.parsers import parse
from ipa.observability.trace_log import TraceLog, make_event, _hash_text


@dataclass
class FastPathResult:
    artifact_id: str
    mime_type: str
    parser_id: str
    document_id: str | None
    pages: int
    chunks_created: int
    first_queryable: bool
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)


class FastPathRunner:
    """Orchestrates the fast path pipeline over a Landing Zone."""

    def __init__(
        self,
        landing_db: str | Path,
        store_db: str | Path,
        index_db: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        landing_root: str | Path | None = None,
        trace_log: TraceLog | None = None,
        index_backend: str = "bm25",
        quarantine_config: QuarantineConfig | None = None,
        chunker: str = "fixed",
        chunker_threshold: float = 0.3,
        chunker_min_size: int = 500,
        chunker_max_size: int = 3000,
        embedding_adapter=None,
    ) -> None:
        self.landing = LandingZone(landing_db, root=landing_root)
        self.store = DocumentStore(store_db)
        # Index backend: 'bm25' (FTS5) or 'tantivy' (E6 winner)
        self.index_backend = index_backend
        if index_backend == "tantivy":
            from ipa.indexes.tantivy_index import TantivyIndex
            # Tantivy uses a directory, not a single DB file
            self.index = TantivyIndex(index_db)
        else:
            self.index = BM25Index(index_db)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.trace = trace_log
        # Layer 2: quarantine config (None = skip quarantine)
        self.quarantine_config = quarantine_config
        # Chunker selection: 'fixed' (default, fast) or 'semantic' (BGE-M3)
        self.chunker = chunker
        self.chunker_threshold = chunker_threshold
        self.chunker_min_size = chunker_min_size
        self.chunker_max_size = chunker_max_size
        self._embedding_adapter = embedding_adapter
        # Lazy-load embedding adapter for semantic chunker
        if chunker == "semantic" and embedding_adapter is None:
            from ipa.indexes.embedding_adapter import EmbeddingAdapter
            self._embedding_adapter = EmbeddingAdapter(batch_size=16, show_progress=False)
            self._owns_embedding = True
        else:
            self._owns_embedding = False

    def close(self) -> None:
        self.landing.close()
        self.store.close()
        self.index.close()
        if self._owns_embedding and self._embedding_adapter is not None:
            self._embedding_adapter.close()

    def __enter__(self) -> "FastPathRunner":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ingest(self, path: str | Path) -> FastPathResult:
        """Ingest a single file through the full fast path."""
        start = time.monotonic()
        path = Path(path)

        # 1. Landing Zone â€” register artifact (commit here for idempotency)
        t_stage = time.monotonic()
        ref = self.landing.register(path)
        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "landing", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                output_hash=ref.content_hash,
                metadata={"original_filename": ref.original_filename, "byte_size": ref.byte_size},
            ))

        # 2. MIME routing â€” batch landing updates, single commit at end
        t_stage = time.monotonic()
        self.landing.set_status(ref.artifact_id, "accepted")
        mime_type = detect_mime(path)
        self.landing.set_mime_type(ref.artifact_id, mime_type)
        parser_id = route_to_parser(mime_type)
        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "mime", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                metadata={"mime_type": mime_type, "parser_id": parser_id},
            ))
        self.landing.set_stage(ref.artifact_id, "parsing", "running")
        self.landing.set_status(ref.artifact_id, "parsing")

        # Layer 2: Quarantine â€” validate file before parsing
        parse_path = path
        q_path = None
        if self.quarantine_config is not None:
            try:
                ftype = detect_file_type_from_path(path)
                q_path = quarantine_file(path, expected_type=ftype,
                                         config=self.quarantine_config)
                parse_path = q_path
            except QuarantineError as exc:
                self.landing.set_stage(ref.artifact_id, "parsing", "failed")
                self.landing.set_status(ref.artifact_id, "failed")
                self.landing.commit()
                if self.trace:
                    self.trace.emit(make_event(
                        ref.artifact_id, "quarantine", "failed",
                        latency_ms=0, error=str(exc),
                    ))
                return FastPathResult(
                    artifact_id=ref.artifact_id, mime_type=mime_type,
                    parser_id=parser_id, document_id=None, pages=0,
                    chunks_created=0, first_queryable=False,
                    elapsed_seconds=time.monotonic() - start,
                    errors=[f"quarantine failed: {exc}"],
                )

        # 3. Parse
        t_stage = time.monotonic()
        try:
            result = parse(parse_path, ref.artifact_id, parser_id)
        except Exception as exc:
            self.landing.set_stage(ref.artifact_id, "parsing", "failed")
            self.landing.set_status(ref.artifact_id, "failed")
            self.landing.commit()
            if self.trace:
                self.trace.emit(make_event(
                    ref.artifact_id, "parsing", "failed",
                    latency_ms=(time.monotonic() - t_stage) * 1000,
                    error=str(exc),
                ))
            return FastPathResult(
                artifact_id=ref.artifact_id, mime_type=mime_type,
                parser_id=parser_id, document_id=None, pages=0,
                chunks_created=0, first_queryable=False,
                elapsed_seconds=time.monotonic() - start,
                errors=[f"parser {parser_id} failed: {exc}"],
            )
        if result.status != "parsed" or result.canonical_document is None:
            self.landing.set_stage(ref.artifact_id, "parsing", "failed")
            self.landing.set_status(ref.artifact_id, "failed")
            self.landing.commit()
            if self.trace:
                self.trace.emit(make_event(
                    ref.artifact_id, "parsing", "failed",
                    latency_ms=(time.monotonic() - t_stage) * 1000,
                    error=f"parser returned status={result.status}",
                ))
            return FastPathResult(
                artifact_id=ref.artifact_id,
                mime_type=mime_type,
                parser_id=parser_id,
                document_id=None,
                pages=0,
                chunks_created=0,
                first_queryable=False,
                elapsed_seconds=time.monotonic() - start,
                errors=[f"parser {parser_id} returned status={result.status}"],
            )
        doc = result.canonical_document
        self.landing.set_stage(ref.artifact_id, "parsing", "success")
        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "parsing", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                output_hash=_hash_text(doc.text[:4096]),
                metadata={"parser_id": parser_id, "pages": doc.pages,
                          "document_id": doc.document_id},
            ))

        # 4. Chunk
        t_stage = time.monotonic()
        self.landing.set_stage(ref.artifact_id, "chunking", "running")
        if self.chunker == "semantic":
            from ipa.ingestion.alt_chunkers import chunk_document_semantic
            chunks = chunk_document_semantic(
                doc,
                threshold=self.chunker_threshold,
                min_chunk_size=self.chunker_min_size,
                max_chunk_size=self.chunker_max_size,
                embedding_adapter=self._embedding_adapter,
            )
        else:
            chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
        self.landing.set_stage(ref.artifact_id, "chunking", "success")
        if not chunks:
            # Parse succeeded but produced no usable text (scanned PDF whose
            # OCR returned nothing, empty/whitespace files). Store nothing —
            # an empty document would only reach curation to be rejected
            # anyway — and mark the artifact `no_text` so the sweep deletes
            # the file instead of it lingering as "indexed" forever.
            self.landing.set_status(ref.artifact_id, "no_text")
            self.landing.commit()
            if self.trace:
                self.trace.emit(make_event(
                    ref.artifact_id, "pipeline", "no_text",
                    latency_ms=(time.monotonic() - t_stage) * 1000,
                    metadata={"document_id": doc.document_id, "pages": doc.pages},
                ))
            return FastPathResult(
                artifact_id=ref.artifact_id,
                mime_type=mime_type,
                parser_id=parser_id,
                document_id=doc.document_id,
                pages=doc.pages,
                chunks_created=0,
                first_queryable=self.index.is_queryable(),
                elapsed_seconds=time.monotonic() - start,
            )
        self.landing.set_status(ref.artifact_id, "chunked")
        if self.trace:
            chunk_hashes = [_hash_text(c.text) for c in chunks[:10]]
            self.trace.emit(make_event(
                ref.artifact_id, "chunking", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                input_hash=_hash_text(doc.text[:4096]),
                metadata={"chunk_count": len(chunks), "chunk_size": self.chunk_size,
                          "chunker": self.chunker,
                          "sample_hashes": chunk_hashes},
            ))

        # 5. Store â€” single commit for doc + chunks
        t_stage = time.monotonic()
        self.store.put_document(doc, ref.artifact_id)
        self.store.put_chunks(chunks)
        self.store.commit()
        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "storing", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                metadata={"document_id": doc.document_id, "chunk_count": len(chunks)},
            ))

        # 6. Index â€” BM25 / FTS5 â€” single batch commit
        t_stage = time.monotonic()
        self.landing.set_stage(ref.artifact_id, "bm25", "running")
        self.index.add_chunks(chunks)
        self.landing.set_stage(ref.artifact_id, "bm25", "success")
        self.landing.set_status(ref.artifact_id, "indexed")
        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "indexing", "success",
                latency_ms=(time.monotonic() - t_stage) * 1000,
                metadata={"backend": self.index_backend, "chunk_count": len(chunks)},
            ))

        # 7. Flush all landing updates in one commit
        self.landing.commit()

        # 8. first_queryable
        first_queryable = self.index.is_queryable()
        elapsed = time.monotonic() - start

        if self.trace:
            self.trace.emit(make_event(
                ref.artifact_id, "pipeline", "success",
                latency_ms=elapsed * 1000,
                metadata={"first_queryable": first_queryable, "total_chunks": len(chunks)},
            ))

        return FastPathResult(
            artifact_id=ref.artifact_id,
            mime_type=mime_type,
            parser_id=parser_id,
            document_id=doc.document_id,
            pages=doc.pages,
            chunks_created=len(chunks),
            first_queryable=first_queryable,
            elapsed_seconds=elapsed,
        )

    def ingest_directory(
        self,
        input_dir: str | Path | None = None,
        progress: bool = True,
        skip_indexed: bool = True,
    ) -> list[FastPathResult]:
        """Ingest all files in the configured or supplied Landing directory.

        If skip_indexed is True, artifacts already in the Landing Zone with
        status='indexed' are skipped (resume after interruption).
        """
        if input_dir is None:
            paths = list(self.landing.iter_files())
        else:
            # Skip SQLite temp files, scrape history, and other non-content files
            _skip_suffixes = {".db", ".db-journal", ".db-wal", ".db-shm", ".lock", ".tmp"}
            _skip_names = {"scrape_history.db", "scrape_report.json", "scrape_history.db-journal"}
            paths = sorted(
                p for p in Path(input_dir).rglob("*")
                if p.is_file()
                and not p.name.startswith(".")
                and p.name not in _skip_names
                and not any(p.name.endswith(s) for s in _skip_suffixes)
            )
        total = len(paths)
        results: list[FastPathResult] = []
        skipped = 0
        t0 = time.monotonic()
        for i, path in enumerate(paths, 1):
            # Skip already-indexed artifacts (resume after interruption).
            if skip_indexed:
                import hashlib
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                artifact_id = f"sha256:{digest.hexdigest()}"
                status = self.landing.get_status(artifact_id)
                if status in ("indexed", "no_text"):
                    skipped += 1
                    if progress and (skipped % 50 == 0 or skipped == 1):
                        safe_name = path.name[:60].encode("ascii", "replace").decode("ascii")
                        print(f"[{i:>5}/{total}] SKIP {safe_name:<60} ({status}, {skipped} skipped)", flush=True)
                    continue
            result = self.ingest(path)
            results.append(result)
            if progress:
                processed = i - skipped
                elapsed = time.monotonic() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total - i
                eta = remaining / rate if rate > 0 else 0
                status = "OK" if not result.errors else "ERR"
                # Sanitize filename for console encoding (Windows cp1252).
                safe_name = path.name[:60].encode("ascii", "replace").decode("ascii")
                print(
                    f"[{i:>5}/{total}] {status} {safe_name:<60} "
                    f"chunks={result.chunks_created:<5} "
                    f"{result.elapsed_seconds:.2f}s  "
                    f"({rate:.1f}/s  ETA {eta:.0f}s)",
                    flush=True,
                )
        if progress and skipped > 0:
            print(f"\nSkipped {skipped} already-indexed artifacts. Processed {len(results)} new.")
        return results

    def search(self, query: str, limit: int = 10):
        """Search the BM25 index. Delegates to BM25Index.search."""
        return self.index.search(query, limit)

