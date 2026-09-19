"""MCP server tests — search_knowledge rerank path + import safety.

The MCP server had two latent bugs with no safety net (no MCP suite):
  1. sys.path shadowing: inserting src/ipa (the package dir itself) made
     `ipa/mcp/` shadow the `mcp` SDK on sys.path → the module could not
     even be imported ("No module named 'mcp.server'").
  2. RerankCandidate(id=...) — the dataclass field is chunk_id; the
     TypeError was swallowed by the fallback except, so search_knowledge
     silently ran WITHOUT reranking.
"""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.indexes.reranker_adapter import RerankCandidate  # noqa: E402
from ipa.mcp import mcp_server  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes — the real components load BGE-M3 / reranker / LanceDB (heavy).
# ---------------------------------------------------------------------------

class _FakeEmbed:
    def embed_query_hybrid(self, query: str):
        return [0.1, 0.2, 0.3], {"1": 0.5}


class _FakeHit:
    def __init__(self, chunk_id: str, score: float):
        self.chunk_id = chunk_id
        self.score = score


class _FakeLance:
    def __init__(self, hits):
        self._hits = hits

    def search_hybrid(self, query, dense, *, query_sparse=None, limit=10):
        return self._hits[:limit]


class _FakeStore:
    """DocumentStore stand-in: get_chunk + the _conn surface _availability uses."""

    def __init__(self, chunks: dict):
        self._chunks = chunks
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE chunks (tombstoned INTEGER DEFAULT 0, text TEXT)")
        self._conn.execute("CREATE TABLE embedding_jobs (status TEXT)")

    def get_chunk(self, chunk_id):
        return self._chunks.get(chunk_id)


class _FakeReranker:
    """Records candidates; returns them REVERSED — an observable ordering."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[list[str]] = []
        self._error = error

    def rerank(self, query, candidates, top_k=5, normalize=True):
        if self._error:
            raise self._error
        self.calls.append([c.chunk_id for c in candidates])
        return [RerankCandidate(
            chunk_id=c.chunk_id, text=c.text, score=float(len(candidates) - i),
            metadata=c.metadata,
        ) for i, c in enumerate(reversed(candidates))][:top_k]


def _wire(monkeypatch, *, hits, chunks, reranker):
    store = _FakeStore(chunks)
    monkeypatch.setattr(mcp_server._ComponentCache, "_embedding_adapter", _FakeEmbed())
    monkeypatch.setattr(mcp_server._ComponentCache, "_lancedb_index", _FakeLance(hits))
    monkeypatch.setattr(mcp_server._ComponentCache, "_document_store", store)
    monkeypatch.setattr(mcp_server._ComponentCache, "_reranker", reranker)


def _default_hits():
    return [_FakeHit(f"chunk:{i}", 1.0 - i * 0.1) for i in range(6)]


def _default_chunks():
    return {
        f"chunk:{i}": SimpleNamespace(
            text=f"texto {i}", document_id=f"doc:{i}", source_span=None,
        )
        for i in range(6)
    }


# ---------------------------------------------------------------------------

def test_mcp_server_imports_and_exposes_tools():
    """Guards the sys.path fix: inserting src/ipa (not src/) made `ipa/mcp/`
    shadow the `mcp` SDK and the module failed to import entirely."""
    assert callable(mcp_server.search_knowledge)
    assert callable(mcp_server.ingest_url)
    assert callable(mcp_server.list_sources)


def test_search_knowledge_applies_reranker(monkeypatch):
    """El reranker se invoca y su orden gana — con el bug latente
    (RerankCandidate(id=...)) el TypeError se tragaba y el output era el
    orden híbrido sin rerankear."""
    chunks = {f"chunk:{i}": SimpleNamespace(
        text=f"texto {i}", document_id=f"doc:{i}", source_span=None)
        for i in range(6)}
    reranker = _FakeReranker()
    _wire(monkeypatch, hits=_default_hits(), chunks=chunks, reranker=reranker)

    out = json.loads(mcp_server.search_knowledge("consulta", top_k=3))

    assert len(reranker.calls) == 1  # el reranker SÍ se invocó
    # Los candidatos llegan con el campo correcto (chunk_id, no id).
    assert reranker.calls[0] == [f"chunk:{i}" for i in range(6)]
    # El orden de salida es el del reranker (reverso), no el híbrido.
    assert [r["chunk_id"] for r in out["results"]] == ["chunk:5", "chunk:4", "chunk:3"]
    assert out["total"] == 3
    assert all("document_id" in r for r in out["results"])


def test_search_knowledge_falls_back_when_reranker_fails(monkeypatch):
    """Si el reranker falla, el fallback devuelve los hits híbridos sin
    rerankear — la búsqueda nunca se rompe por el stage-2."""
    chunks = {f"chunk:{i}": SimpleNamespace(text=f"texto {i}", document_id=f"doc:{i}", source_span=None)
              for i in range(6)}
    reranker = _FakeReranker(error="boom")
    _wire(monkeypatch, hits=_default_hits(), chunks=chunks, reranker=reranker)

    out = json.loads(mcp_server.search_knowledge("consulta", top_k=3))
    assert out["total"] == 3
    assert [r["chunk_id"] for r in out["results"]] == [f"chunk:{i}" for i in range(3)]
