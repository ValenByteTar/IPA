"""Adaptive re-chunking â€” post-processing for low-quality chunks.

After the initial fixed_window chunking, some documents produce chunks
with low lexical density (tables, TOCs, number lists cut in half).  This
module detects those documents and re-processes them:

  1. MERGE: combine adjacent low-density chunks from the same document
     into a single larger chunk.  This often restores enough context
     for the chunk to be retrievable (e.g., a full table instead of
     half a table).

  2. FALLBACK RE-CHUNK: if the merged chunk still has low lexical density,
     re-chunk the document's text with the recursive chunker, which
     respects natural paragraph and section boundaries.

Activation criteria:
  - Document has >30% of chunks with lexical_density < 0.4
  - Individual chunks with ld < 0.2 are always candidates for merge

This is a deterministic post-processing step â€” no LLM required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan
from ipa.ingestion.chunker import _chunk_id, _content_hash, _map_span


# ---------------------------------------------------------------------------
# Lexical density (same as analysis scripts, kept here for self-containment)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
})


def lexical_density(text: str) -> float:
    """Compute lexical density: unique content words / total content words.

    Returns 0.0 for text with no content words (pure numbers, symbols).
    """
    tokens = re.findall(r'[a-zA-Z]{2,}', text.lower())
    content = [t for t in tokens if t not in _STOPWORDS]
    if not content:
        return 0.0
    return len(set(content)) / len(content)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RechunkResult:
    """Result of adaptive re-chunking for a single document."""
    document_id: str
    original_chunk_count: int
    final_chunk_count: int
    merged_count: int
    fallback_rechunk_count: int
    chunks: list[DocumentChunk] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def adaptive_rechunk(
    doc: CanonicalDocument,
    chunks: list[DocumentChunk],
    merge_threshold: float = 0.4,
    always_merge_threshold: float = 0.2,
    doc_problem_ratio: float = 0.30,
    merge_max_size: int = 2048,
    recursive_chunk_size: int = 800,
    recursive_overlap: int = 100,
) -> RechunkResult:
    """Adaptively re-chunk a document if it has quality issues.

    Args:
        doc: The canonical document.
        chunks: Original chunks (from fixed_window or any chunker).
        merge_threshold: Chunks with ld below this are merge candidates.
        always_merge_threshold: Chunks with ld below this are always merged
            (even if the document doesn't meet the problem ratio).
        doc_problem_ratio: If >this fraction of chunks have ld < merge_threshold,
            the document is flagged for re-chunking.
        merge_max_size: Don't merge beyond this size (chars).
        recursive_chunk_size: Chunk size for fallback recursive re-chunking.
        recursive_overlap: Overlap for fallback recursive re-chunking.

    Returns:
        RechunkResult with the final chunk list and statistics.
    """
    if not chunks:
        return RechunkResult(
            document_id=doc.document_id,
            original_chunk_count=0,
            final_chunk_count=0,
            merged_count=0,
            fallback_rechunk_count=0,
        )

    # Step 1: Compute lexical density for each chunk.
    chunk_lds = [lexical_density(c.text) for c in chunks]
    problem_chunks = sum(1 for ld in chunk_lds if ld < merge_threshold)
    problem_ratio = problem_chunks / len(chunks)

    # Also check for always-merge chunks (ld < always_merge_threshold).
    has_always_merge = any(ld < always_merge_threshold for ld in chunk_lds)

    # Step 2: Decide if re-chunking is needed.
    needs_rechunk = (problem_ratio > doc_problem_ratio) or has_always_merge

    if not needs_rechunk:
        return RechunkResult(
            document_id=doc.document_id,
            original_chunk_count=len(chunks),
            final_chunk_count=len(chunks),
            merged_count=0,
            fallback_rechunk_count=0,
            chunks=chunks,
            reason="no_action",
        )

    # Step 3: MERGE â€” combine adjacent low-density chunks.
    merged_chunks, merged_count = _merge_adjacent_low_density(
        chunks, chunk_lds, merge_threshold, always_merge_threshold, merge_max_size
    )

    # Step 4: Check if merged chunks are still problematic.
    merged_lds = [lexical_density(c.text) for c in merged_chunks]
    still_problem = sum(1 for ld in merged_lds if ld < merge_threshold)
    still_problem_ratio = still_problem / len(merged_chunks) if merged_chunks else 0

    # Step 5: FALLBACK RE-CHUNK if still problematic.
    if still_problem_ratio > doc_problem_ratio:
        final_chunks, fallback_rechunk_count = _fallback_rechunk(
            doc, merged_chunks, merged_lds, merge_threshold,
            recursive_chunk_size, recursive_overlap,
        )
        reason = f"merge+rechunke (problem_ratio={problem_ratio:.2f}, still={still_problem_ratio:.2f})"
    else:
        final_chunks = merged_chunks
        fallback_rechunk_count = 0
        reason = f"merge_only (problem_ratio={problem_ratio:.2f}, merged={merged_count})"

    return RechunkResult(
        document_id=doc.document_id,
        original_chunk_count=len(chunks),
        final_chunk_count=len(final_chunks),
        merged_count=merged_count,
        fallback_rechunk_count=fallback_rechunk_count,
        chunks=final_chunks,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Merge adjacent low-density chunks
# ---------------------------------------------------------------------------

def _merge_adjacent_low_density(
    chunks: list[DocumentChunk],
    chunk_lds: list[float],
    merge_threshold: float,
    always_merge_threshold: float,
    max_size: int,
) -> tuple[list[DocumentChunk], int]:
    """Merge adjacent chunks that both have low lexical density.

    Only merges if:
      - Both chunks have ld < merge_threshold, OR
      - One chunk has ld < always_merge_threshold (force merge with neighbor)

    Stops merging when combined size exceeds max_size.
    """
    if len(chunks) <= 1:
        return chunks, 0

    result: list[DocumentChunk] = []
    merge_count = 0
    i = 0

    while i < len(chunks):
        current = chunks[i]
        current_ld = chunk_lds[i]
        current_text = current.text
        current_size = len(current_text)

        # Try to merge with next chunks while both are low density.
        j = i + 1
        while j < len(chunks):
            next_chunk = chunks[j]
            next_ld = chunk_lds[j]
            next_size = len(next_chunk.text)

            # Don't exceed max size.
            if current_size + next_size > max_size:
                break

            # Merge condition: both low density, or current is very low.
            should_merge = (
                (current_ld < merge_threshold and next_ld < merge_threshold)
                or (current_ld < always_merge_threshold)
                or (next_ld < always_merge_threshold and current_ld < merge_threshold)
            )

            if not should_merge:
                break

            # Merge: combine text and metadata.
            current_text = current_text + "\n" + next_chunk.text
            current_size = len(current_text)
            current_ld = lexical_density(current_text)  # recompute after merge
            merge_count += 1
            j += 1

        if j > i + 1:
            # We merged at least one chunk â€” create a new merged chunk.
            merged = DocumentChunk(
                chunk_id=_chunk_id(current.document_id, len(result)),
                document_id=current.document_id,
                content_hash=_content_hash(current_text),
                text=current_text,
                metadata={
                    **current.metadata,
                    "rechunked": True,
                    "merge_count": j - i - 1,
                    "original_chunker": current.metadata.get("splitter", "fixed_window"),
                    "new_chunker": "adaptive_merge",
                },
                source_span=current.source_span,
            )
            result.append(merged)
        else:
            # No merge â€” keep original chunk but update index.
            if len(result) != i:
                # Re-index the chunk_id to match new position.
                result.append(DocumentChunk(
                    chunk_id=_chunk_id(current.document_id, len(result)),
                    document_id=current.document_id,
                    content_hash=current.content_hash,
                    text=current.text,
                    metadata=current.metadata,
                    source_span=current.source_span,
                ))
            else:
                result.append(current)

        i = j

    return result, merge_count


# ---------------------------------------------------------------------------
# Fallback re-chunk with recursive splitter
# ---------------------------------------------------------------------------

def _fallback_rechunk(
    doc: CanonicalDocument,
    merged_chunks: list[DocumentChunk],
    merged_lds: list[float],
    merge_threshold: float,
    chunk_size: int,
    overlap: int,
) -> tuple[list[DocumentChunk], int]:
    """Re-chunk the document text with recursive splitter.

    Only re-chunks if the merged chunks are still low quality.
    Uses the recursive chunker (LangChain) which respects paragraph
    and section boundaries.
    """
    from ipa.ingestion.alt_chunkers import chunk_document_recursive

    # Re-chunk the entire document text.
    new_chunks = chunk_document_recursive(
        doc, chunk_size=chunk_size, overlap=overlap
    )

    # Mark as re-chunked.
    for i, chunk in enumerate(new_chunks):
        chunk.metadata["rechunked"] = True
        chunk.metadata["original_chunker"] = "fixed_window"
        chunk.metadata["new_chunker"] = "recursive_fallback"

    fallback_rechunk_count = len(new_chunks)
    return new_chunks, fallback_rechunk_count


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def adaptive_rechunk_batch(
    docs_and_chunks: list[tuple[CanonicalDocument, list[DocumentChunk]]],
    **kwargs,
) -> list[RechunkResult]:
    """Process multiple documents and return re-chunking results.

    Args:
        docs_and_chunks: List of (document, chunks) pairs.
        **kwargs: Passed to adaptive_rechunk.

    Returns:
        List of RechunkResult, one per document.
    """
    results = []
    for doc, chunks in docs_and_chunks:
        result = adaptive_rechunk(doc, chunks, **kwargs)
        results.append(result)
    return results

