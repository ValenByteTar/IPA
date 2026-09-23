"""Tests for Stage 1 fast path components and end-to-end integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipa import (
    BM25Index,
    CanonicalDocument,
    DocumentChunk,
    DocumentStore,
    FastPathRunner,
    LandingZone,
    SourceSpan,
    chunk_document,
    detect_mime,
    parse,
    route_to_parser,
)


# --- Landing Zone ---

class TestLandingZone:
    def test_register_single_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        with LandingZone(tmp_path / "landing.db") as lz:
            ref = lz.register(f)
            assert ref.artifact_id.startswith("sha256:")
            assert ref.content_hash == ref.artifact_id
            assert ref.original_filename == "a.txt"
            assert ref.byte_size == 5
            assert lz.count() == 1

    def test_register_is_idempotent(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        with LandingZone(tmp_path / "landing.db") as lz:
            ref1 = lz.register(f)
            ref2 = lz.register(f)
            assert ref1.artifact_id == ref2.artifact_id
            assert lz.count() == 1

    def test_set_and_get_stage(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        with LandingZone(tmp_path / "landing.db") as lz:
            ref = lz.register(f)
            assert lz.get_stage(ref.artifact_id, "parsing") is None
            lz.set_stage(ref.artifact_id, "parsing", "success")
            lz.set_status(ref.artifact_id, "indexed")
            assert lz.get_stage(ref.artifact_id, "parsing") == "success"
            assert lz.get_status(ref.artifact_id) == "indexed"

    def test_export_manifest(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        with LandingZone(tmp_path / "landing.db") as lz:
            ref = lz.register(f)
            lz.set_mime_type(ref.artifact_id, "text/plain")
            lz.set_status(ref.artifact_id, "indexed")
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_hash = lz.export_manifest(manifest_path)
            assert manifest_hash.startswith("sha256:")
            content = manifest_path.read_text(encoding="utf-8")
            assert "a.txt" in content
            assert "text/plain" in content
            assert '"status": "indexed"' in content

    def test_configured_root_discovers_files(self, tmp_path):
        landing = tmp_path / "Landing"
        landing.mkdir()
        (landing / "a.txt").write_text("hello", encoding="utf-8")
        (landing / ".gitkeep").write_text("", encoding="utf-8")
        with LandingZone(tmp_path / "landing.db", root=landing) as lz:
            paths = list(lz.iter_files())
            refs = lz.register_directory()
            assert paths == [landing / "a.txt"]
            assert len(refs) == 1
            assert lz.count() == 1


# --- MIME Router ---

class TestMimeRouter:
    def test_detect_txt(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello", encoding="utf-8")
        assert detect_mime(f) == "text/plain"

    def test_detect_html(self, tmp_path):
        f = tmp_path / "file.html"
        f.write_text("<html></html>", encoding="utf-8")
        assert detect_mime(f) == "text/html"

    def test_detect_json(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("{}", encoding="utf-8")
        assert detect_mime(f) == "application/json"

    def test_detect_pdf_by_magic_bytes(self, tmp_path):
        f = tmp_path / "noext"
        f.write_bytes(b"%PDF-1.4 fake pdf content")
        assert detect_mime(f) == "application/pdf"

    def test_content_signature_overrides_wrong_extension(self, tmp_path):
        f = tmp_path / "renamed.txt"
        f.write_bytes(b"%PDF-1.7 fake pdf content")
        assert detect_mime(f) == "application/pdf"

    def test_detect_unknown_binary(self, tmp_path):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        assert detect_mime(f) == "application/octet-stream"

    def test_route_pdf_to_pymupdf(self):
        assert route_to_parser("application/pdf") == "pymupdf"

    def test_route_html(self):
        assert route_to_parser("text/html") == "html"

    def test_route_text(self):
        assert route_to_parser("text/plain") == "text"

    def test_route_json(self):
        assert route_to_parser("application/json") == "json"

    def test_route_unknown(self):
        assert route_to_parser("application/octet-stream") == "unknown"


# --- Parsers ---

class TestParsers:
    def test_parse_text(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("Hello world.\nLine two.", encoding="utf-8")
        result = parse(f, "sha256:abc", "text")
        assert result.status == "parsed"
        assert result.canonical_document is not None
        assert "Hello world" in result.canonical_document.text
        assert result.canonical_document.parser_id == "text"
        assert len(result.canonical_document.source_spans) == 1

    def test_parse_html_strips_tags(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><h1>Title</h1><p>Body</p></body></html>", encoding="utf-8")
        result = parse(f, "sha256:abc", "html")
        assert result.status == "parsed"
        text = result.canonical_document.text
        assert "<" not in text
        assert "Title" in text
        assert "Body" in text

    def test_parse_json_with_text_field(self, tmp_path):
        """JSON with a 'text' field extracts it as content, rest as metadata."""
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps({"text": "The main content here.", "source_file": "x.pdf", "pages": 5}),
            encoding="utf-8",
        )
        result = parse(f, "sha256:abc", "json")
        assert result.status == "parsed"
        doc = result.canonical_document
        assert "The main content here." in doc.text
        # Metadata keys preserved in elements, not in text.
        assert "source_file" not in doc.text
        assert "pages" not in doc.text
        assert doc.elements[0]["keys"] == ["source_file", "pages"]
        assert doc.elements[0]["metadata"]["pages"] == 5

    def test_parse_json_without_text_field_falls_back(self, tmp_path):
        """JSON without a 'text' field falls back to re-serialization."""
        f = tmp_path / "data.json"
        f.write_text('{"key": "value", "n": 42}', encoding="utf-8")
        result = parse(f, "sha256:abc", "json")
        assert result.status == "parsed"
        assert "key" in result.canonical_document.text
        assert "value" in result.canonical_document.text

    def test_parse_pdf(self, tmp_path):
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content", fontsize=12)
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()
        result = parse(pdf_path, "sha256:abc", "pymupdf")
        assert result.status == "parsed"
        assert result.canonical_document is not None
        assert "Test PDF content" in result.canonical_document.text
        assert result.canonical_document.pages == 1
        assert len(result.canonical_document.source_spans) == 1

    def test_parse_unknown_parser_returns_failed(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello", encoding="utf-8")
        result = parse(f, "sha256:abc", "nonexistent")
        assert result.status == "failed"
        assert result.canonical_document is None

    # --- Text normalization tests ---

    def test_normalize_removes_pua_chars(self, tmp_path):
        """Private-Use Area glyphs (Wingdings) are removed."""
        f = tmp_path / "note.txt"
        f.write_text("Hello \uf0a7 world \uf0b0 end", encoding="utf-8")
        result = parse(f, "sha256:abc", "text")
        text = result.canonical_document.text
        assert "\uf0a7" not in text
        assert "\uf0b0" not in text
        assert "Hello" in text and "world" in text and "end" in text

    def test_normalize_removes_replacement_char(self, tmp_path):
        """U+FFFD replacement characters are removed."""
        f = tmp_path / "note.txt"
        f.write_text("Copyright \ufffd 2010 Cisco", encoding="utf-8")
        result = parse(f, "sha256:abc", "text")
        assert "\ufffd" not in result.canonical_document.text
        assert "Copyright" in result.canonical_document.text

    def test_normalize_collapses_whitespace(self, tmp_path):
        """Excessive whitespace is collapsed."""
        f = tmp_path / "note.txt"
        f.write_text("Line 1\n\n\n\n\n\nLine 2     spaced", encoding="utf-8")
        result = parse(f, "sha256:abc", "text")
        text = result.canonical_document.text
        assert "\n\n\n" not in text
        assert "     " not in text
        assert "Line 1" in text and "Line 2" in text

    def test_html_parser_strips_comments(self, tmp_path):
        """HTML comments are removed."""
        f = tmp_path / "page.html"
        f.write_text("<!-- todo: fix --><p>visible</p>", encoding="utf-8")
        result = parse(f, "sha256:abc", "html")
        assert "todo" not in result.canonical_document.text
        assert "visible" in result.canonical_document.text


# --- Chunker ---

class TestChunker:
    def _make_doc(self, text: str = "A" * 1000) -> CanonicalDocument:
        return CanonicalDocument(
            document_id="doc:test",
            pages=1,
            elements=[{"page": 1, "char_count": len(text)}],
            source_spans=[SourceSpan(
                artifact_id="sha256:abc", page=1,
                offset_start=0, offset_end=len(text),
            )],
            text=text,
            mime_type="text/plain",
            parser_id="text",
        )

    def test_chunk_deterministic_ids(self):
        doc = self._make_doc()
        chunks1 = chunk_document(doc, chunk_size=100, overlap=20)
        chunks2 = chunk_document(doc, chunk_size=100, overlap=20)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_content_hashes_match_text(self):
        doc = self._make_doc("Hello world. " * 50)
        chunks = chunk_document(doc, chunk_size=100, overlap=20)
        for chunk in chunks:
            expected = "sha256:" + hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            assert chunk.content_hash == expected

    def test_chunk_overlap_coverage(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        # Every character should be covered by at least one chunk.
        covered = set()
        for chunk in chunks:
            for i in range(chunk.metadata["char_start"], chunk.metadata["char_end"]):
                covered.add(i)
        for i in range(len(text)):
            assert i in covered, f"char {i} not covered"

    def test_chunk_empty_text_returns_empty(self):
        doc = self._make_doc("")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_chunk_invalid_params_raise(self):
        doc = self._make_doc()
        with pytest.raises(ValueError):
            chunk_document(doc, chunk_size=0)
        with pytest.raises(ValueError):
            chunk_document(doc, chunk_size=100, overlap=100)

    def test_chunk_source_span_mapped(self):
        doc = self._make_doc("A" * 300)
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.source_span is not None
            assert chunk.source_span.artifact_id == "sha256:abc"
            assert chunk.source_span.page == 1


# --- DocumentStore ---

class TestDocumentStore:
    def _make_doc(self) -> CanonicalDocument:
        return CanonicalDocument(
            document_id="doc:test1",
            pages=2,
            elements=[{"page": 1}, {"page": 2}],
            source_spans=[SourceSpan("sha256:abc", 1, 0, 100)],
            text="Test document text.",
            mime_type="text/plain",
            parser_id="text",
        )

    def test_put_and_get_document(self, tmp_path):
        with DocumentStore(tmp_path / "store.db") as store:
            doc = self._make_doc()
            store.put_document(doc, "sha256:abc")
            retrieved = store.get_document("doc:test1")
            assert retrieved is not None
            assert retrieved.text == "Test document text."
            assert retrieved.pages == 2
            assert len(retrieved.source_spans) == 1
            assert retrieved.source_spans[0].artifact_id == "sha256:abc"

    def test_put_and_get_chunks(self, tmp_path):
        with DocumentStore(tmp_path / "store.db") as store:
            doc = self._make_doc()
            store.put_document(doc, "sha256:abc")
            chunks = [
                DocumentChunk(
                    chunk_id="chunk:c1", document_id="doc:test1",
                    content_hash="sha256:x", text="chunk text 1",
                    metadata={"index": 0},
                    source_span=SourceSpan("sha256:abc", 1, 0, 10),
                ),
                DocumentChunk(
                    chunk_id="chunk:c2", document_id="doc:test1",
                    content_hash="sha256:y", text="chunk text 2",
                    metadata={"index": 1},
                    source_span=SourceSpan("sha256:abc", 1, 10, 20),
                ),
            ]
            store.put_chunks(chunks)
            assert store.count_chunks() == 2
            retrieved = list(store.get_chunks("doc:test1"))
            assert len(retrieved) == 2
            assert retrieved[0].text == "chunk text 1"
            assert retrieved[0].source_span is not None
            assert retrieved[0].source_span.page == 1

    def test_tombstone_document(self, tmp_path):
        with DocumentStore(tmp_path / "store.db") as store:
            doc = self._make_doc()
            store.put_document(doc, "sha256:abc")
            store.put_chunks([DocumentChunk(
                chunk_id="chunk:c1", document_id="doc:test1",
                content_hash="sha256:x", text="text",
                metadata={},
            )])
            store.tombstone_document("doc:test1")
            assert store.count_documents() == 0
            assert store.count_chunks() == 0
            assert store.get_document("doc:test1") is None


# --- BM25Index ---

class TestBM25Index:
    def _make_chunk(self, chunk_id: str, text: str, doc_id: str = "doc:1") -> DocumentChunk:
        return DocumentChunk(
            chunk_id=chunk_id, document_id=doc_id,
            content_hash="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            text=text, metadata={},
            source_span=SourceSpan("sha256:abc", 1, 0, len(text)),
        )

    def test_add_and_search(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            idx.add_chunk(self._make_chunk("c1", "The quick brown fox jumps over the lazy dog"))
            idx.add_chunk(self._make_chunk("c2", "Machine learning models require training data"))
            assert idx.count() == 2
            assert idx.is_queryable()
            hits = idx.search("fox")
            assert len(hits) >= 1
            assert hits[0].chunk_id == "c1"
            assert hits[0].retrieval_backend == "sqlite_fts5"
            assert hits[0].source_span is not None

    def test_search_no_results(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            idx.add_chunk(self._make_chunk("c1", "hello world"))
            hits = idx.search("nonexistent_term_xyz")
            assert hits == []

    def test_search_sanitizes_fts_syntax(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            idx.add_chunk(self._make_chunk("c1", "hello world"))
            assert idx.search('hello "')
            with pytest.raises(ValueError):
                idx.search("hello", limit=0)

    def test_remove_chunk_tombstone(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            idx.add_chunk(self._make_chunk("c1", "hello world"))
            assert idx.count() == 1
            idx.remove_chunk("c1")
            assert idx.count() == 0
            assert not idx.is_queryable()

    def test_add_chunk_is_idempotent(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            chunk = self._make_chunk("c1", "hello world")
            idx.add_chunk(chunk)
            idx.add_chunk(chunk)
            assert idx.count() == 1

    def test_search_provenance(self, tmp_path):
        with BM25Index(tmp_path / "index.db") as idx:
            idx.add_chunk(self._make_chunk("c1", "contract compliance test"))
            hits = idx.search("contract")
            assert len(hits) == 1
            hit = hits[0]
            assert hit.source_span is not None
            assert hit.source_span.artifact_id == "sha256:abc"
            assert hit.source_span.page == 1


# --- Fast Path Integration ---

class TestFastPathIntegration:
    def test_ingest_text_file_end_to_end(self, tmp_path):
        f = tmp_path / "input" / "note.txt"
        f.parent.mkdir(parents=True)
        f.write_text("RES-023 lab ingestion test. " * 20, encoding="utf-8")
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            result = runner.ingest(f)
            assert result.artifact_id.startswith("sha256:")
            assert result.mime_type == "text/plain"
            assert result.parser_id == "text"
            assert result.document_id is not None
            assert result.chunks_created > 0
            assert result.first_queryable is True
            assert result.errors == []
            assert runner.landing.get_status(result.artifact_id) == "indexed"

    def test_ingest_directory_multiple_formats(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "note.txt").write_text("Plain text content for testing. " * 10, encoding="utf-8")
        (input_dir / "page.html").write_text("<html><body><p>HTML content here</p></body></html>", encoding="utf-8")
        (input_dir / "data.json").write_text('{"key": "value", "list": [1, 2, 3]}', encoding="utf-8")

        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            results = runner.ingest_directory(input_dir)
            assert len(results) == 3
            assert all(r.first_queryable for r in results)
            assert all(r.errors == [] for r in results)
            mime_types = {r.mime_type for r in results}
            assert "text/plain" in mime_types
            assert "text/html" in mime_types
            assert "application/json" in mime_types

    def test_ingest_then_search(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "doc1.txt").write_text(
            "The RES-023 lab builds a continuous ingestion hopper for personal AGI.", encoding="utf-8"
        )
        (input_dir / "doc2.txt").write_text(
            "Machine learning requires clean training data and good evaluation metrics.", encoding="utf-8"
        )
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            runner.ingest_directory(input_dir)
            hits = runner.search("ingestion hopper")
            assert len(hits) >= 1
            # The hit should come from doc1 which contains both terms.
            doc1_id = next(
                ref.artifact_id for ref in runner.landing.list_artifacts()
                if ref.original_filename == "doc1.txt"
            )
            assert any(
                hit.source_span is not None
                and hit.source_span.artifact_id == doc1_id
                for hit in hits
            )

    def test_ingest_pdf_end_to_end(self, tmp_path):
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PDF fast path test content", fontsize=12)
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Second page with more text", fontsize=12)
        pdf_path = tmp_path / "input" / "test.pdf"
        pdf_path.parent.mkdir(parents=True)
        doc.save(str(pdf_path))
        doc.close()

        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            result = runner.ingest(pdf_path)
            assert result.mime_type == "application/pdf"
            assert result.parser_id == "pymupdf"
            assert result.pages == 2
            assert result.chunks_created > 0
            assert result.first_queryable is True

    def test_configured_landing_root_can_run_without_input_argument(self, tmp_path):
        landing = tmp_path / "Landing"
        landing.mkdir()
        (landing / "note.txt").write_text("Configured Landing root content.", encoding="utf-8")
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
            landing_root=landing,
        ) as runner:
            results = runner.ingest_directory()
            assert len(results) == 1
            assert results[0].first_queryable is True

    def test_ingest_empty_text_marks_no_text(self, tmp_path):
        """Parse OK pero 0 chunks (texto vacío) → status no_text, sin doc."""
        f = tmp_path / "input" / "blank.txt"
        f.parent.mkdir(parents=True)
        f.write_text("   \n\n   \t  ", encoding="utf-8")
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            result = runner.ingest(f)
            assert result.chunks_created == 0
            assert result.errors == []
            assert runner.landing.get_status(result.artifact_id) == "no_text"
            # Nada entra al store: un doc sin texto solo sería rechazado después.
            assert runner.store.count_documents() == 0
            assert runner.store.count_chunks() == 0

    def test_ingest_directory_skips_no_text_on_rerun(self, tmp_path):
        """Re-run de la misma carpeta no reprocesa artefactos no_text."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "blank.txt").write_text("   ", encoding="utf-8")
        (input_dir / "note.txt").write_text("Real content here. " * 10, encoding="utf-8")
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            first = runner.ingest_directory(input_dir, progress=False)
            assert len(first) == 2
            second = runner.ingest_directory(input_dir, progress=False)
            # Ambos ya resueltos: indexed + no_text → nada se reprocesa.
            assert second == []

    def test_ingest_is_idempotent(self, tmp_path):
        f = tmp_path / "input" / "note.txt"
        f.parent.mkdir(parents=True)
        f.write_text("Idempotency test content. " * 10, encoding="utf-8")
        with FastPathRunner(
            landing_db=tmp_path / "landing.db",
            store_db=tmp_path / "store.db",
            index_db=tmp_path / "index.db",
        ) as runner:
            r1 = runner.ingest(f)
            r2 = runner.ingest(f)
            assert r1.artifact_id == r2.artifact_id
            assert r1.document_id == r2.document_id
            # Chunks should be the same count (idempotent)
            assert r1.chunks_created == r2.chunks_created


# ---------------------------------------------------------------------------
# PM-004: drain de embeddings acotado por pasada + lock de trabajos pesados
# ---------------------------------------------------------------------------

class _DrainChunk:
    def __init__(self, chunk_id: str, document_id: str = "doc:1"):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = "texto"


class _DrainStore:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def all_chunks(self):
        return list(self._chunks)

    def count_chunks(self):
        return len(self._chunks)

    def close(self):
        pass


class _DrainArrowTable:
    def __init__(self, ids):
        self._ids = list(ids)

    def to_pylist(self):
        return [{"chunk_id": chunk_id} for chunk_id in self._ids]


class _DrainTable:
    def __init__(self, ids):
        self._ids = list(ids)

    def to_arrow(self):
        return _DrainArrowTable(self._ids)


class _DrainLance:
    def __init__(self):
        self.added: list[str] = []
        self._table = None

    def is_queryable(self):
        return self._table is not None

    def add_chunks(self, chunks, vectors, sparse_weights=None):
        self.added.extend(c.chunk_id for c in chunks)
        self._table = _DrainTable(self.added)

    def close(self):
        pass


class _DrainEmbed:
    def __init__(self):
        self.calls = 0
        self.device = "cpu"
        self.closed = False
        self.assert_gpu_lease = False

    def try_move_to_gpu(self):
        self.device = "cuda"
        return True

    def embed_texts_hybrid(self, texts):
        if self.assert_gpu_lease:
            from ipa.providers import vram_lock
            holder = vram_lock.holder()
            assert holder is not None and holder["owner"] == "bulk_embedding"
            assert self.device == "cuda"
        self.calls += 1
        return [[0.0, 0.0] for _ in texts], [{} for _ in texts]

    def close(self):
        self.closed = True
        self.device = "closed"


def test_index_lancedb_incremental_respects_max_chunks():
    """La pasada se acota: antes la primera pasada recorría el corpus entero y
    el watch reportaba '0 chunks embedded' durante horas (PM-004)."""
    from ipa.ingestion.fast_path_cli import _index_lancedb_incremental

    chunks = [_DrainChunk(f"c{i}") for i in range(10)]
    store, lance, embed = _DrainStore(chunks), _DrainLance(), _DrainEmbed()
    result = _index_lancedb_incremental(store, lance, embed, batch_size=2,
                                        max_chunks=4)
    assert result["new_chunks"] == 4
    assert len(lance.added) == 4

    # Segunda pasada con el mismo known_ids: sigue desde donde quedó.
    known = set(lance.added)
    result2 = _index_lancedb_incremental(store, lance, embed, batch_size=2,
                                         known_ids=known, max_chunks=4)
    assert result2["new_chunks"] == 4
    assert len(lance.added) == 8


def test_embed_drain_loop_yields_to_interactive_waiter(tmp_path, monkeypatch):
    """El drain (background) cede el lock a un waiter interactivo y lo retoma
    cuando el interactivo termina — el escenario del incidente (PM-004)."""
    import threading
    import time as _time

    from ipa.agentic import embedding_maintenance, heavy_lock
    from ipa.ingestion import fast_path_cli

    monkeypatch.setattr(heavy_lock, "LOCK_PATH", tmp_path / "heavy.lock")
    monkeypatch.setattr(heavy_lock, "WAIT_PATH", tmp_path / "heavy.waiting")
    # Hermetic: claim_job/renew/release touch the real embedding_maintenance
    # lock/state under outputs/web_dashboard — a live dashboard's idle
    # scheduler can legitimately hold it and starve the test's drain.
    monkeypatch.setattr(embedding_maintenance, "JOB_LOCK_PATH",
                        tmp_path / "embedding_maintenance.lock")
    monkeypatch.setattr(embedding_maintenance, "STATE_PATH",
                        tmp_path / "embedding_maintenance.json")

    chunks = [_DrainChunk(f"c{i}") for i in range(4)]
    store, lance, embed = _DrainStore(chunks), _DrainLance(), _DrainEmbed()
    monkeypatch.setattr("ipa.DocumentStore", lambda *a, **k: store)
    monkeypatch.setattr("ipa.indexes.embedding_adapter.EmbeddingAdapter",
                        lambda *a, **k: embed)
    monkeypatch.setattr("ipa.indexes.lancedb_index.LanceDBIndex",
                        lambda *a, **k: lance)

    # Un research está esperando: el drain no debe embeder.
    heavy_lock.register_waiter("research", heavy_lock.PRIORITY_INTERACTIVE)
    done = threading.Event()
    stats = {"indexed": 0, "idle": False}
    thread = threading.Thread(
        target=fast_path_cli._embed_drain_loop,
        args=(tmp_path / "document_store.db", tmp_path / "lancedb", done, stats),
        kwargs={"batch_size": 2}, daemon=True,
    )
    thread.start()
    _time.sleep(1.0)
    assert stats["indexed"] == 0, "el drain no puede embeder con un waiter interactivo"
    assert lance.added == []

    # El interactivo termina: el drain retoma y drena todo.
    heavy_lock.unregister_waiter()
    deadline = _time.monotonic() + 12
    while stats["indexed"] < len(chunks) and _time.monotonic() < deadline:
        _time.sleep(0.2)
    assert stats["indexed"] == len(chunks)
    assert sorted(lance.added) == ["c0", "c1", "c2", "c3"]

    done.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert heavy_lock.holder() is None


def test_bulk_drain_switches_to_gpu_at_threshold_and_resumes_from_lance(
        tmp_path, monkeypatch):
    """El job mantiene BGE en GPU todo el drain, restaura chat y al re-run
    omite los chunk_ids ya confirmados en LanceDB."""
    import threading

    from ipa.agentic import embedding_maintenance, heavy_lock
    from ipa.ingestion import fast_path_cli
    from ipa.providers import ollama_provider, vram_lock

    monkeypatch.setattr(vram_lock, "LOCK_PATH", tmp_path / "vram.lock")
    monkeypatch.setattr(vram_lock, "_pid_alive_cache", {})
    monkeypatch.setattr(embedding_maintenance, "STATE_PATH", tmp_path / "embed.json")
    monkeypatch.setattr(embedding_maintenance, "JOB_LOCK_PATH", tmp_path / "embed.lock")
    monkeypatch.setattr(heavy_lock, "LOCK_PATH", tmp_path / "heavy.lock")
    monkeypatch.setattr(heavy_lock, "WAIT_PATH", tmp_path / "heavy.waiting")
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_BULK_ENABLED", True)
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_MIN_BACKLOG", 512)
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_WAIT_SECONDS", 1.0)
    model = {"name": "qwen3.5:9b-q4_K_M"}
    monkeypatch.setattr(ollama_provider, "loaded_ollama_models", lambda: [model])
    monkeypatch.setattr(ollama_provider, "unload_ollama_models", lambda timeout_s=20: [model["name"]])
    warmups = []
    monkeypatch.setattr(ollama_provider, "warmup_ollama_model",
                        lambda name, keep_alive="30m": warmups.append(name))
    monkeypatch.setattr("ipa.indexes.lancedb_index._compute_centroids",
                        lambda store, lance: None)

    chunks = [_DrainChunk(f"c{i}") for i in range(520)]
    store, lance, embed = _DrainStore(chunks), _DrainLance(), _DrainEmbed()
    embed.assert_gpu_lease = True
    monkeypatch.setattr("ipa.DocumentStore", lambda *a, **k: store)
    monkeypatch.setattr("ipa.indexes.embedding_adapter.EmbeddingAdapter",
                        lambda *a, **k: embed)
    monkeypatch.setattr("ipa.indexes.lancedb_index.LanceDBIndex",
                        lambda *a, **k: lance)

    done = threading.Event()
    done.set()  # corpus ya no recibe nuevos chunks; ir directo al final drain
    stats = {"indexed": 0, "idle": False}
    fast_path_cli._embed_drain_loop(
        tmp_path / "document_store.db", tmp_path / "lancedb", done, stats,
        batch_size=64,
    )

    assert stats["indexed"] == len(chunks)
    assert set(lance.added) == {f"c{i}" for i in range(len(chunks))}
    assert embed.closed is True
    assert warmups == [model["name"]]
    assert vram_lock.holder() is None
    assert embedding_maintenance.read_state()["status"] == "completed"
    assert embedding_maintenance.read_state()["chat_blocked"] is False

    # Reinicio/resume: los IDs ya guardados no vuelven a pasar por el embedder.
    second_embed = _DrainEmbed()
    second_embed.assert_gpu_lease = True
    monkeypatch.setattr("ipa.indexes.embedding_adapter.EmbeddingAdapter",
                        lambda *a, **k: second_embed)
    second_stats = {"indexed": 0, "idle": False}
    fast_path_cli._embed_drain_loop(
        tmp_path / "document_store.db", tmp_path / "lancedb", done, second_stats,
        batch_size=64,
    )
    assert second_stats["indexed"] == 0
    assert second_embed.calls == 0
    assert warmups == [model["name"]]


def _bulk_module(monkeypatch, tmp_path, *, loaded_models=None):
    """Prepara el lote GPU con locks y estado aislados en tmp_path."""
    from ipa.agentic import embedding_maintenance
    from ipa.ingestion import fast_path_cli
    from ipa.providers import ollama_provider, vram_lock

    monkeypatch.setattr(vram_lock, "LOCK_PATH", tmp_path / "vram.lock")
    monkeypatch.setattr(vram_lock, "_pid_alive_cache", {})
    monkeypatch.setattr(embedding_maintenance, "STATE_PATH", tmp_path / "embed.json")
    monkeypatch.setattr(embedding_maintenance, "JOB_LOCK_PATH", tmp_path / "embed.lock")
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_BULK_ENABLED", True)
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_MIN_BACKLOG", 512)
    monkeypatch.setattr(fast_path_cli, "EMBED_GPU_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(ollama_provider, "loaded_ollama_models",
                        lambda: loaded_models or [])
    warmups = []
    unloads = []
    monkeypatch.setattr(ollama_provider, "unload_ollama_models",
                        lambda timeout_s=20: unloads.append(timeout_s) or [m["name"] for m in (loaded_models or [])])
    monkeypatch.setattr(ollama_provider, "warmup_ollama_model",
                        lambda model, keep_alive="30m": warmups.append((model, keep_alive)))
    return fast_path_cli, embedding_maintenance, vram_lock, warmups, unloads


class _BulkEmbedding:
    def __init__(self, upgrade_ok=True):
        self.upgrade_ok = upgrade_ok
        self.upgrades = 0
        self.closes = 0

    def try_move_to_gpu(self):
        self.upgrades += 1
        return self.upgrade_ok

    def close(self):
        self.closes += 1


def test_gpu_bulk_threshold_is_512_and_lower_backlog_stays_cpu(tmp_path, monkeypatch):
    fp, maintenance, vram_lock, warmups, unloads = _bulk_module(monkeypatch, tmp_path)
    embedding = _BulkEmbedding()
    assert fp._start_bulk_gpu(embedding, corpus=tmp_path, total=2000,
                              vectorized=1489, pending=511) is None
    assert embedding.upgrades == 0
    assert vram_lock.holder() is None
    assert not maintenance.STATE_PATH.exists()


def test_gpu_bulk_keeps_bge_and_chat_lock_until_finalization(tmp_path, monkeypatch):
    model = {"name": "qwen3.5:9b-q4_K_M"}
    fp, maintenance, vram_lock, warmups, unloads = _bulk_module(
        monkeypatch, tmp_path, loaded_models=[model])
    embedding = _BulkEmbedding()

    session = fp._start_bulk_gpu(
        embedding, corpus=tmp_path, total=2000, vectorized=1488, pending=512)
    assert session["active"] is True
    assert embedding.upgrades == 1
    assert unloads == [20.0]
    assert vram_lock.holder()["owner"] == "bulk_embedding"
    assert vram_lock.acquire("ollama") is False
    state = maintenance.read_state()
    assert state["status"] == "embedding"
    assert state["chat_blocked"] is True
    assert embedding.closes == 0  # BGE no se descarga entre batches

    fp._update_bulk_gpu(session, embedded=128, total=2000,
                        vectorized_before=1488, pending=384)
    state = maintenance.read_state()
    assert state["embedded"] == 128
    assert state["pending"] == 384
    assert vram_lock.holder()["owner"] == "bulk_embedding"  # no unload per pass

    fp._finish_bulk_gpu(embedding, session, status="completed")
    assert embedding.closes == 1
    assert warmups == [("qwen3.5:9b-q4_K_M", "30m")]
    assert vram_lock.holder() is None
    assert maintenance.read_state()["chat_blocked"] is False


def test_gpu_bulk_falls_back_to_cpu_and_restores_chat_if_cuda_fails(
        tmp_path, monkeypatch):
    model = {"name": "qwen3.5:9b-q4_K_M"}
    fp, maintenance, vram_lock, warmups, unloads = _bulk_module(
        monkeypatch, tmp_path, loaded_models=[model])
    embedding = _BulkEmbedding(upgrade_ok=False)

    session = fp._start_bulk_gpu(
        embedding, corpus=tmp_path, total=2000, vectorized=0, pending=2000)
    assert session is None
    assert embedding.closes == 0  # conservar/usar el modelo CPU del drain
    assert unloads == [20.0]
    assert warmups == [("qwen3.5:9b-q4_K_M", "30m")]
    assert vram_lock.holder() is None
    state = maintenance.read_state()
    assert state["status"] == "cpu_fallback"
    assert state["chat_blocked"] is False


def test_gpu_bulk_respects_other_motor_lock(tmp_path, monkeypatch):
    fp, maintenance, vram_lock, _, _ = _bulk_module(monkeypatch, tmp_path)
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: True)
    vram_lock.LOCK_PATH.write_text("424242|exl3|9999999999", encoding="utf-8")
    embedding = _BulkEmbedding()

    session = fp._start_bulk_gpu(
        embedding, corpus=tmp_path, total=2000, vectorized=0, pending=2000)
    assert session is None
    assert embedding.upgrades == 0
    assert vram_lock.holder()["owner"] == "exl3"
    assert maintenance.read_state()["status"] == "cpu_fallback"
    assert maintenance.read_state()["chat_blocked"] is False


def test_gpu_bulk_cancellation_releases_lock_and_restores_llm(tmp_path, monkeypatch):
    model = {"name": "qwen3.5:9b-q4_K_M"}
    fp, maintenance, vram_lock, warmups, _ = _bulk_module(
        monkeypatch, tmp_path, loaded_models=[model])
    embedding = _BulkEmbedding()
    session = fp._start_bulk_gpu(
        embedding, corpus=tmp_path, total=2000, vectorized=0, pending=2000)

    fp._finish_bulk_gpu(embedding, session, status="cancelled",
                        error="cancelled at a batch boundary")
    assert vram_lock.holder() is None
    assert warmups
    assert maintenance.read_state()["status"] == "cancelled"
    assert maintenance.read_state()["chat_blocked"] is False


def test_gpu_bulk_lease_times_out_waiting_for_exl3(tmp_path, monkeypatch):
    fp, maintenance, vram_lock, _, _ = _bulk_module(monkeypatch, tmp_path)
    monkeypatch.setattr(vram_lock, "pid_alive", lambda pid: True)
    vram_lock.LOCK_PATH.write_text("424242|exl3|9999999999", encoding="utf-8")
    embedding = _BulkEmbedding()

    session = fp._start_bulk_gpu(
        embedding, corpus=tmp_path, total=2000, vectorized=0, pending=2000)
    assert session is None
    assert embedding.upgrades == 0
    assert vram_lock.holder()["owner"] == "exl3"
    assert maintenance.read_state()["status"] == "cpu_fallback"
    assert maintenance.read_state()["chat_blocked"] is False
