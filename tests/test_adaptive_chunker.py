"""Tests for adaptive re-chunking module.

Validates:
  - Lexical density computation
  - Merge of adjacent low-density chunks
  - Fallback re-chunk when merge is insufficient
  - No-action when chunks are already good quality
  - Batch processing
  - Metadata marking on re-chunked chunks
"""
from __future__ import annotations

import pytest

from ipa import CanonicalDocument, DocumentChunk, SourceSpan
from ipa.ingestion.adaptive_chunker import (
    adaptive_rechunk,
    adaptive_rechunk_batch,
    lexical_density,
    RechunkResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "doc:test") -> CanonicalDocument:
    return CanonicalDocument(
        document_id=doc_id,
        pages=1,
        elements=[{"page": 1, "char_count": len(text)}],
        source_spans=[SourceSpan(
            artifact_id="sha256:test",
            page=1,
            offset_start=0,
            offset_end=len(text),
        )],
        text=text,
        mime_type="text/plain",
        parser_id="text",
    )


def _make_chunk(doc_id: str, idx: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk:{doc_id}:{idx}",
        document_id=doc_id,
        content_hash=f"sha256:{idx}",
        text=text,
        metadata={"chunk_index": idx, "splitter": "fixed_window"},
        source_span=SourceSpan(
            artifact_id="sha256:test",
            page=1,
            offset_start=idx * 512,
            offset_end=idx * 512 + len(text),
        ),
    )


# ---------------------------------------------------------------------------
# Lexical density tests
# ---------------------------------------------------------------------------

class TestLexicalDensity:
    def test_normal_text(self):
        text = "The router forwards packets between networks using routing tables."
        ld = lexical_density(text)
        assert 0.5 < ld <= 1.0  # should have decent density

    def test_pure_numbers(self):
        text = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
        ld = lexical_density(text)
        assert ld == 0.0  # no content words

    def test_table_data(self):
        text = "255.255.248.0 /21 2,048 0.0.7.255 /20 255.255.240.0 4,096"
        ld = lexical_density(text)
        assert ld == 0.0  # no content words

    def test_empty(self):
        assert lexical_density("") == 0.0

    def test_only_stopwords(self):
        assert lexical_density("the a an and or but in on at to for") == 0.0

    def test_high_density(self):
        text = "BGP OSPF routing protocol autonomous systems border gateways"
        ld = lexical_density(text)
        assert ld == 1.0  # all unique content words


# ---------------------------------------------------------------------------
# No-action tests
# ---------------------------------------------------------------------------

class TestNoAction:
    def test_good_chunks_not_modified(self):
        """Documents with high-density chunks should not be re-chunked."""
        doc = _make_doc("BGP routing protocol. OSPF internal routing. DNS resolution. DHCP configuration.")
        chunks = [
            _make_chunk("doc:good", 0, "BGP routing protocol for autonomous systems."),
            _make_chunk("doc:good", 1, "OSPF internal routing within autonomous systems."),
            _make_chunk("doc:good", 2, "DNS resolution and DHCP configuration management."),
        ]
        result = adaptive_rechunk(doc, chunks)
        assert result.reason == "no_action"
        assert result.merged_count == 0
        assert result.fallback_rechunk_count == 0
        assert result.final_chunk_count == len(chunks)
        # Chunks should be the same objects (not modified).
        assert result.chunks == chunks

    def test_single_chunk_no_action(self):
        """A single chunk should not be re-chunked."""
        doc = _make_doc("Some text here.")
        chunks = [_make_chunk("doc:single", 0, "Some text here.")]
        result = adaptive_rechunk(doc, chunks)
        assert result.reason == "no_action"
        assert result.final_chunk_count == 1

    def test_empty_chunks(self):
        doc = _make_doc("")
        result = adaptive_rechunk(doc, [])
        assert result.original_chunk_count == 0
        assert result.final_chunk_count == 0


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_adjacent_low_density(self):
        """Two adjacent low-density chunks should be merged."""
        doc = _make_doc("1 2 3 4 5 6 7 8 9 10\nBGP routing protocol\n11 12 13 14 15 16 17 18 19 20")
        chunks = [
            _make_chunk("doc:merge", 0, "1 2 3 4 5 6 7 8 9 10"),  # ld=0
            _make_chunk("doc:merge", 1, "BGP routing protocol for autonomous systems"),  # ld high
            _make_chunk("doc:merge", 2, "11 12 13 14 15 16 17 18 19 20"),  # ld=0
        ]
        result = adaptive_rechunk(doc, chunks)
        assert result.merged_count >= 1
        assert result.reason.startswith("merge")

    def test_merge_respects_max_size(self):
        """Merging should stop when max_size is exceeded."""
        # Create low-density chunks that are individually small but
        # would exceed max_size when merged.
        chunks = [
            _make_chunk("doc:big", i, f"{i} {i+1} {i+2} {i+3} {i+4} {i+5} {i+6} {i+7} {i+8} {i+9}")
            for i in range(0, 200, 10)
        ]
        doc = _make_doc(" ".join(c.text for c in chunks))
        # max_size=30 means at most ~2-3 chunks can merge before hitting limit.
        result = adaptive_rechunk(doc, chunks, merge_max_size=30)
        # The merge phase must not exceed max_size; fallback may then
        # legitimately produce a single recursive chunk for numeric text.
        assert result.merged_count == 0
        assert result.final_chunk_count <= len(chunks)

    def test_merged_chunk_has_metadata(self):
        """Merged chunks should be marked with rechunked=True."""
        doc = _make_doc("1 2 3 4 5\n6 7 8 9 10\nBGP routing protocol")
        chunks = [
            _make_chunk("doc:meta", 0, "1 2 3 4 5"),
            _make_chunk("doc:meta", 1, "6 7 8 9 10"),
            _make_chunk("doc:meta", 2, "BGP routing protocol for autonomous systems"),
        ]
        result = adaptive_rechunk(doc, chunks)
        merged = [c for c in result.chunks if c.metadata.get("rechunked")]
        assert len(merged) > 0
        assert merged[0].metadata["new_chunker"] == "adaptive_merge"

    def test_does_not_merge_high_density_neighbors(self):
        """High-density chunks should not be merged with each other."""
        doc = _make_doc("BGP routing. OSPF routing. DNS resolution. DHCP config.")
        chunks = [
            _make_chunk("doc:high", 0, "BGP routing protocol for autonomous systems."),
            _make_chunk("doc:high", 1, "OSPF internal routing within networks."),
        ]
        result = adaptive_rechunk(doc, chunks)
        assert result.merged_count == 0
        assert result.reason == "no_action"


# ---------------------------------------------------------------------------
# Always-merge threshold tests
# ---------------------------------------------------------------------------

class TestAlwaysMerge:
    def test_always_merge_extremely_low_density(self):
        """Chunks with ld < always_merge_threshold should always be merged."""
        doc = _make_doc("1 2 3 4 5 6 7 8 9 10\nBGP routing protocol for networks")
        chunks = [
            _make_chunk("doc:always", 0, "1 2 3 4 5 6 7 8 9 10"),  # ld=0
            _make_chunk("doc:always", 1, "BGP routing protocol for autonomous networks."),  # ld high
        ]
        # With only 1/2 = 50% problem chunks, but ld=0 triggers always_merge.
        result = adaptive_rechunk(doc, chunks, always_merge_threshold=0.2)
        assert result.merged_count >= 1


# ---------------------------------------------------------------------------
# Batch processing tests
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_batch_mixed_docs(self):
        """Batch should process some docs and skip others."""
        good_doc = _make_doc("BGP routing protocol. OSPF internal routing.")
        good_chunks = [
            _make_chunk("doc:good", 0, "BGP routing protocol for autonomous systems."),
            _make_chunk("doc:good", 1, "OSPF internal routing within networks."),
        ]
        bad_doc = _make_doc("1 2 3 4 5\n6 7 8 9 10\nBGP routing")
        bad_chunks = [
            _make_chunk("doc:bad", 0, "1 2 3 4 5 6 7 8 9 10"),
            _make_chunk("doc:bad", 1, "11 12 13 14 15 16 17 18 19 20"),
            _make_chunk("doc:bad", 2, "BGP routing protocol for networks."),
        ]

        results = adaptive_rechunk_batch([
            (good_doc, good_chunks),
            (bad_doc, bad_chunks),
        ])

        assert len(results) == 2
        assert results[0].reason == "no_action"
        assert results[1].merged_count > 0

    def test_batch_empty(self):
        results = adaptive_rechunk_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Integration with existing chunkers
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_rechunk_preserves_document_id(self):
        """Re-chunked chunks should keep the original document_id."""
        doc = _make_doc("1 2 3 4 5\n6 7 8 9 10\nBGP routing protocol")
        chunks = [
            _make_chunk("doc:preserve", 0, "1 2 3 4 5 6 7 8 9 10"),
            _make_chunk("doc:preserve", 1, "11 12 13 14 15 16 17 18 19 20"),
            _make_chunk("doc:preserve", 2, "BGP routing protocol for networks."),
        ]
        result = adaptive_rechunk(doc, chunks)
        for chunk in result.chunks:
            assert chunk.document_id == "doc:preserve"

    def test_rechunk_produces_valid_chunks(self):
        """All output chunks should be valid DocumentChunk objects."""
        doc = _make_doc("1 2 3 4 5\n6 7 8 9 10\nBGP routing protocol")
        chunks = [
            _make_chunk("doc:valid", 0, "1 2 3 4 5 6 7 8 9 10"),
            _make_chunk("doc:valid", 1, "11 12 13 14 15 16 17 18 19 20"),
            _make_chunk("doc:valid", 2, "BGP routing protocol for networks."),
        ]
        result = adaptive_rechunk(doc, chunks)
        for chunk in result.chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.chunk_id.startswith("chunk:")
            assert chunk.content_hash.startswith("sha256:")
            assert len(chunk.text) > 0
