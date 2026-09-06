"""Alternative chunker adapters for E5 â€” chunking competition.

Candidates:
  - LangChain RecursiveCharacterTextSplitter (sentence-aware splitting)
  - LangChain TokenTextSplitter (token-based splitting)
  - Semantic chunker (embedding-based boundary detection)

All produce the same DocumentChunk records as the baseline chunker,
enabling direct comparison of boundary quality, duplicate rate, and
retrieval recall.
"""
from __future__ import annotations

from typing import Any

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan
from ipa.ingestion.chunker import _chunk_id, _content_hash, _map_span


def chunk_document_recursive(
    doc: CanonicalDocument,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Split using LangChain RecursiveCharacterTextSplitter.

    Tries to split on separators: ["\n\n", "\n", ". ", " ", ""] in order.
    Produces more natural boundaries than fixed-window.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    texts = splitter.split_text(doc.text)
    chunks: list[DocumentChunk] = []

    # Map character offsets back to source spans.
    search_pos = 0
    for index, chunk_text in enumerate(texts):
        # Find this chunk's position in the original text.
        pos = doc.text.find(chunk_text[:100], search_pos)
        if pos == -1:
            pos = search_pos
        end = pos + len(chunk_text)
        span = _map_span(pos, end, doc.source_spans)
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(doc.document_id, index),
            document_id=doc.document_id,
            content_hash=_content_hash(chunk_text),
            text=chunk_text,
            metadata={
                "chunk_index": index,
                "char_start": pos,
                "char_end": end,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "splitter": "recursive",
            },
            source_span=span,
        ))
        search_pos = end - overlap if end > overlap else end

    return chunks


def chunk_document_token(
    doc: CanonicalDocument,
    chunk_size: int = 200,
    overlap: int = 20,
) -> list[DocumentChunk]:
    """Split using LangChain TokenTextSplitter (tiktoken-based).

    Splits by token count rather than character count, producing more
    consistent chunk sizes for LLM consumption.
    """
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    texts = splitter.split_text(doc.text)
    chunks: list[DocumentChunk] = []

    search_pos = 0
    for index, chunk_text in enumerate(texts):
        pos = doc.text.find(chunk_text[:100], search_pos)
        if pos == -1:
            pos = search_pos
        end = pos + len(chunk_text)
        span = _map_span(pos, end, doc.source_spans)
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(doc.document_id, index),
            document_id=doc.document_id,
            content_hash=_content_hash(chunk_text),
            text=chunk_text,
            metadata={
                "chunk_index": index,
                "char_start": pos,
                "char_end": end,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "splitter": "token",
            },
            source_span=span,
        ))
        search_pos = end

    return chunks


def chunk_document_semantic(
    doc: CanonicalDocument,
    threshold: float = 0.3,
    min_chunk_size: int = 500,
    max_chunk_size: int = 3000,
    embedding_adapter: Any = None,
) -> list[DocumentChunk]:
    """Split using semantic similarity between sentences.

    Groups consecutive sentences until semantic similarity drops below
    threshold, then starts a new chunk.  Produces topically coherent chunks.

    Args:
        threshold: Cosine similarity below which a boundary is created.
            Lower values create fewer, larger chunks.  Based on analysis of
            the corpus, p25=0.31, p50=0.58.  Default 0.3 splits only on
            significant topic shifts.
        min_chunk_size: Don't flush a chunk smaller than this (merge forward).
            Prevents tiny chunks from short sentences or repeated headers.
        max_chunk_size: Force a split if accumulated text exceeds this, even
            if similarity is above threshold.  Prevents unbounded chunks.
        embedding_adapter: Optional pre-loaded EmbeddingAdapter to reuse
            across multiple documents.  Avoids reloading BGE-M3 (1.2 GB)
            for each document.  If None, creates a temporary one.

    Uses BGE-M3 for embeddings (1024 dims, 8K context).
    """
    import numpy as np
    from ipa.indexes.embedding_adapter import EmbeddingAdapter

    # Split into sentences first.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', doc.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Embed all sentences â€” reuse adapter if provided, else create temporary.
    if embedding_adapter is not None:
        vectors = embedding_adapter.embed_texts(sentences)
    else:
        emb = EmbeddingAdapter(show_progress=False)
        vectors = emb.embed_texts(sentences)
        emb.close()

    vectors_np = np.array(vectors)

    # Calculate cosine similarity between consecutive sentences.
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # Group sentences into chunks based on semantic similarity.
    # A boundary is created when:
    #   1. similarity < threshold AND current chunk >= min_chunk_size, or
    #   2. current chunk >= max_chunk_size (forced split)
    raw_chunks: list[list[str]] = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine(vectors_np[i - 1], vectors_np[i])
        should_split = False

        # Forced split if chunk is too large.
        if current_len >= max_chunk_size:
            should_split = True
        # Semantic boundary, but only if chunk is big enough.
        elif sim < threshold and current_len >= min_chunk_size:
            should_split = True

        if should_split:
            raw_chunks.append(current_sentences)
            current_sentences = []
            current_len = 0

        current_sentences.append(sentences[i])
        current_len += len(sentences[i]) + 1

    # Flush remaining.
    if current_sentences:
        # Merge tiny trailing chunk into previous if possible.
        previous_len = sum(len(sentence) for sentence in raw_chunks[-1]) + max(0, len(raw_chunks[-1]) - 1) if raw_chunks else 0
        if current_len < min_chunk_size and raw_chunks and previous_len + current_len + 1 <= max_chunk_size:
            raw_chunks[-1].extend(current_sentences)
        else:
            raw_chunks.append(current_sentences)

    # Build DocumentChunk records from raw sentence groups.
    chunks: list[DocumentChunk] = []
    search_pos = 0
    for index, sent_group in enumerate(raw_chunks):
        chunk_text = " ".join(sent_group)
        if not chunk_text.strip():
            continue
        # Find position in original text.
        pos = doc.text.find(chunk_text[:80], search_pos)
        if pos == -1:
            pos = search_pos
        end = pos + len(chunk_text)
        span = _map_span(pos, end, doc.source_spans)
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(doc.document_id, index),
            document_id=doc.document_id,
            content_hash=_content_hash(chunk_text),
            text=chunk_text,
            metadata={
                "chunk_index": index,
                "char_start": pos,
                "char_end": end,
                "splitter": "semantic",
                "threshold": threshold,
                "min_chunk_size": min_chunk_size,
                "max_chunk_size": max_chunk_size,
                "num_sentences": len(sent_group),
            },
            source_span=span,
        ))
        search_pos = end

    return chunks

