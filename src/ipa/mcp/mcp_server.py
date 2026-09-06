"""IPA MCP Server â€” local Model Context Protocol server for agentic consumption.

Exposes the IPA knowledge pipeline as MCP tools that a local LLM agent
(Ollama, Claude, etc.) can invoke via stdio transport.

Tools:
  - search_knowledge(query, top_k=5)
      Hybrid search (dense BGE-M3 + FTS BM25 + reranker) over indexed chunks.
      Returns ranked chunks with provenance.

  - ingest_url(url)
      Fetch a URL (auto: requests â†’ Playwright fallback), extract content
      (site-agnostic ContentFilter), validate safety (3 layers), parse,
      semantic chunk (BGE-M3), embed, and index in LanceDB + FTS.
      Returns document_id and chunk count.

  - ingest_file(path)
      Parse a local file (PDF, HTML, text, JSON), chunk, embed, index.
      Returns document_id and chunk count.

  - list_sources()
      List distinct document sources (domains, file types) in the index.

  - get_document(doc_id)
      Retrieve full document text and metadata by document_id.

Usage (standalone):
  python -m ipa.mcp.mcp_server

Usage (with MCP client config):
  Add to client's MCP config:
  {
    "mcpServers": {
      "ipa": {
        "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
        "args": ["-m", "ipa.mcp.mcp_server"],
        "env": {"PYTHONPATH": "C:\\path\\to\\IPA\\src"}
      }
    }
  }

Architecture:
  All components are local. No external APIs, no cloud services.
  - Embeddings: BGE-M3 (local, FP16 GPU)
  - Vector store: LanceDB (local)
  - Lexical index: FTS5 / BM25 (local)
  - Reranker: bge-reranker-v2-m3 (local)
  - Fetch: requests + Playwright (local)
  - LLM agent: Ollama (local) â€” consumes this MCP server
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

# Ensure src is on the path when run as module
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from mcp.server.fastmcp import FastMCP

from ipa.ingestion.content_filter import ContentFilter
from ipa.ingestion.content_safety import (
    DownloadValidationConfig,
    SafeParseConfig,
    detect_file_type_from_path,
)
from ipa.contracts import DocumentChunk, SourceSpan
from ipa.acquisition.fetch_strategy import AutoFetchStrategy


# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


# Database paths â€” configurable via environment
LANCEDB_DIR = _env_path("IPA_LANCEDB_DIR", "outputs/experiments/E12-corpus/vector/lancedb")
DOCUMENT_STORE = _env_path("IPA_DOC_STORE", "outputs/experiments/E12-corpus/document_store.db")
LANDING_DB = _env_path("IPA_LANDING_DB", "outputs/experiments/E12-corpus/landing.db")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_INBOX = Path(os.environ.get("IPA_MCP_INBOX", PROJECT_ROOT / "Landing" / "mcp"))
TANTIVY_DIR = _env_path("IPA_TANTIVY_DIR", "outputs/experiments/E12-corpus/tantivy_index")
_ALLOWED_DOMAINS = {
    item.strip().lower().lstrip(".")
    for item in os.environ.get("IPA_SCRAPE_ALLOWED_DOMAINS", "").split(",")
    if item.strip()
}

# Model settings
EMBED_BATCH_SIZE = int(os.environ.get("IPA_EMBED_BATCH_SIZE", "16"))
RERANKER_TOP_K = int(os.environ.get("IPA_RERANKER_TOP_K", "5"))
SEARCH_TOP_K = int(os.environ.get("IPA_SEARCH_TOP_K", "10"))


# ---------------------------------------------------------------------------
# Lazy-loaded components (heavy resources loaded on first use)
# ---------------------------------------------------------------------------

class _ComponentCache:
    """Lazy singleton cache for heavy components.

    Components are loaded on first use and reused across tool calls.
    This avoids loading BGE-M3 (2GB) on every search/ingest.
    """

    _embedding_adapter = None
    _reranker = None
    _lancedb_index = None
    _document_store = None
    _content_filter = None
    _fetch_strategy = None
    _semantic_chunker = None

    @classmethod
    def get_embedding_adapter(cls):
        if cls._embedding_adapter is None:
            from ipa.indexes.embedding_adapter import EmbeddingAdapter
            cls._embedding_adapter = EmbeddingAdapter(
                batch_size=EMBED_BATCH_SIZE,
                show_progress=False,
            )
        return cls._embedding_adapter

    @classmethod
    def get_reranker(cls):
        if cls._reranker is None:
            from ipa.indexes.reranker_adapter import RerankerAdapter
            cls._reranker = RerankerAdapter()
        return cls._reranker

    @classmethod
    def get_lancedb_index(cls):
        if cls._lancedb_index is None:
            from ipa.indexes.lancedb_index import LanceDBIndex
            cls._lancedb_index = LanceDBIndex(LANCEDB_DIR)
        return cls._lancedb_index

    @classmethod
    def get_document_store(cls):
        if cls._document_store is None:
            from ipa.storage.document_store import DocumentStore
            cls._document_store = DocumentStore(DOCUMENT_STORE)
        return cls._document_store

    @classmethod
    def get_content_filter(cls):
        if cls._content_filter is None:
            cls._content_filter = ContentFilter()
        return cls._content_filter

    @classmethod
    def get_fetch_strategy(cls):
        if cls._fetch_strategy is None:
            cls._fetch_strategy = AutoFetchStrategy()
        return cls._fetch_strategy

    @classmethod
    def get_semantic_chunker(cls):
        if cls._semantic_chunker is None:
            from ipa.ingestion.alt_chunkers import chunk_document_semantic
            cls._semantic_chunker = chunk_document_semantic
        return cls._semantic_chunker

    @classmethod
    def close_all(cls):
        if cls._embedding_adapter is not None:
            cls._embedding_adapter.close()
            cls._embedding_adapter = None
        if cls._reranker is not None:
            cls._reranker.close()
            cls._reranker = None
        if cls._lancedb_index is not None:
            cls._lancedb_index.close()
            cls._lancedb_index = None
        if cls._document_store is not None:
            cls._document_store.close()
            cls._document_store = None
        if cls._fetch_strategy is not None:
            cls._fetch_strategy.close()
            cls._fetch_strategy = None


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _validate_url(url: str) -> tuple[bool, str | None]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False, "only http/https URLs are allowed"
    host = parsed.hostname.lower().rstrip(".")
    try:
        addresses = socket.getaddrinfo(host, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, "private, loopback, link-local, and reserved hosts are blocked"
    except socket.gaierror:
        return False, "host cannot be resolved"
    if _ALLOWED_DOMAINS and not any(host == d or host.endswith("." + d) for d in _ALLOWED_DOMAINS):
        return False, f"domain not allowed: {host}"
    if not _ALLOWED_DOMAINS:
        return False, "scraping is disabled until IPA_SCRAPE_ALLOWED_DOMAINS is configured"
    return True, None


def _safe_file(path: str) -> tuple[Path | None, str | None]:
    candidate = Path(path).expanduser().resolve()
    inbox = MCP_INBOX.resolve()
    if candidate.is_relative_to(inbox) and candidate.is_file():
        return candidate, None
    return None, f"file must be an existing file inside MCP inbox: {inbox}"


def _availability(store) -> dict:
    conn = store._conn
    total = conn.execute("SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]
    embedded = 0
    try:
        embedded = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status='complete'").fetchone()[0]
    except Exception:
        pass
    enriched = conn.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '[Summary]%' ").fetchone()[0]
    return {
        "lexical": "available",
        "vector": "available" if embedded == total and total else "partial",
        "vector_chunks": embedded,
        "total_chunks": total,
        "enriched_chunks": enriched,
        "enrichment": "available" if enriched else "pending",
        "tier_2": "not_implemented",
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "ipa",
    instructions=(
        "IPA Knowledge Server â€” local RAG pipeline for agentic consumption. "
        "Use search_knowledge to find relevant content. "
        "Use ingest_url to add web pages. "
        "Use ingest_file to add local files. "
        "All processing is local (no external APIs)."
    ),
)


# ---------------------------------------------------------------------------
# Tool: search_knowledge
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant content.

    Uses hybrid retrieval: dense vector search (BGE-M3) + lexical search
    (BM25/FTS) with RRF fusion, then cross-encoder reranking
    (bge-reranker-v2-m3) for final precision.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 5, max 20).

    Returns:
        JSON array of results, each with:
        - chunk_id: unique chunk identifier
        - text: chunk content
        - score: relevance score (higher = better)
        - document_id: parent document ID
        - source: document source URL or filename
        - page: page number if available
    """
    top_k = max(1, min(top_k, 20))

    try:
        emb = _ComponentCache.get_embedding_adapter()
        index = _ComponentCache.get_lancedb_index()
        store = _ComponentCache.get_document_store()

        # Embed query with both dense and learned sparse representations.
        query_vector, query_sparse = emb.embed_query_hybrid(query)

        # Hybrid search remains usable while the vector index is partial.
        hits = index.search_hybrid(query, query_vector, query_sparse=query_sparse, limit=top_k * 3)

        if not hits:
            return json.dumps({"results": [], "query": query, "total": 0})

        # Fetch chunk texts from document store
        candidates = []
        for hit in hits[:top_k * 3]:
            chunk = store.get_chunk(hit.chunk_id)
            if chunk is not None:
                candidates.append({
                    "chunk_id": hit.chunk_id,
                    "text": chunk.text,
                    "score": hit.score,
                    "document_id": chunk.document_id,
                    "page": _extract_page(chunk),
                })

        if not candidates:
            return json.dumps({"results": [], "query": query, "total": 0})

        # Rerank with cross-encoder
        try:
            reranker = _ComponentCache.get_reranker()
            from ipa.indexes.reranker_adapter import RerankCandidate
            rerank_candidates = [
                RerankCandidate(
                    id=c["chunk_id"],
                    text=c["text"],
                    score=c["score"],
                    metadata={"document_id": c["document_id"]},
                )
                for c in candidates
            ]
            reranked = reranker.rerank(query, rerank_candidates, top_k=top_k)

            results = []
            for r in reranked:
                results.append({
                    "chunk_id": r.id,
                    "text": r.text,
                    "score": r.score,
                    "document_id": r.metadata.get("document_id", ""),
                    "page": _extract_page_from_id(r.id),
                })
        except Exception:
            # Reranker failed â€” return hybrid results without reranking
            results = candidates[:top_k]

        return _json({
            "results": results[:top_k],
            "query": query,
            "total": len(results),
            "availability": _availability(store),
        })

    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


# ---------------------------------------------------------------------------
# Tool: ingest_url
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_url(url: str) -> str:
    """Fetch and ingest a web page into the knowledge base.

    Automatically selects the best fetch strategy (requests for static
    HTML, Playwright for JS-rendered sites). Extracts main content with
    a site-agnostic filter (no per-site config needed). Validates safety
    (magic bytes, size limits, structure). Chunks with BGE-M3 semantic
    chunker, embeds, and indexes in LanceDB + FTS.

    Args:
        url: URL to fetch and ingest (http:// or https://).

    Returns:
        JSON with:
        - status: "success" or "error"
        - document_id: unique ID for the ingested document
        - chunks: number of chunks created
        - engine: fetch engine used ("requests" or "playwright")
        - title: page title
        - error: error message if status is "error"
    """
    start = time.monotonic()

    try:
        allowed, reason = _validate_url(url)
        if not allowed:
            return _json({"status": "error", "url": url, "error": reason})
        fetcher = _ComponentCache.get_fetch_strategy()
        cf = _ComponentCache.get_content_filter()

        # Step 1: Fetch
        fetch_result = fetcher.fetch(url, timeout=30)

        if not fetch_result.success:
            return json.dumps({
                "status": "error",
                "url": url,
                "error": fetch_result.error or "fetch failed",
                "engine": fetch_result.engine,
            })

        if not fetch_result.html:
            return json.dumps({
                "status": "error",
                "url": url,
                "error": "no HTML content returned",
                "engine": fetch_result.engine,
            })

        # Step 2: Extract content (site-agnostic)
        extracted = cf.extract_detailed(fetch_result.html, base_url=url)
        text = extracted.text

        if not text or len(text) < 50:
            return json.dumps({
                "status": "error",
                "url": url,
                "error": "extracted content too short or empty",
                "engine": fetch_result.engine,
                "title": extracted.title,
            })

        # Step 3: Create document and chunk
        import hashlib
        doc_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        document_id = f"doc:url-{doc_hash}"

        # Semantic chunk
        chunk_fn = _ComponentCache.get_semantic_chunker()
        emb = _ComponentCache.get_embedding_adapter()
        from ipa.contracts import CanonicalDocument
        canonical = CanonicalDocument(
            document_id=document_id, pages=1,
            elements=[{"url": url, "title": extracted.title}],
            source_spans=[], text=text, mime_type="text/html", parser_id="content_filter",
        )
        chunks = chunk_fn(canonical, embedding_adapter=emb)

        if not chunks:
            return json.dumps({
                "status": "error",
                "url": url,
                "error": "chunking produced no chunks",
            })

        # Step 4: Embed chunks
        chunk_texts = [c.text for c in chunks]
        vectors, sparse_weights = emb.embed_texts_hybrid(chunk_texts)

        # Persist canonical data first; vector indexing is a derived stage.
        store = _ComponentCache.get_document_store()
        store.put_document(canonical, artifact_id=f"url:{doc_hash}")
        store.put_chunks(chunks)
        store.commit()

        # Direct MCP writes are protected by the process-level policy in the
        # deployment; the normal orchestrator consumes the same queue.
        index = _ComponentCache.get_lancedb_index()
        index.add_chunks(chunks, vectors, sparse_weights=sparse_weights)
        index.create_fts_index()

        elapsed = time.monotonic() - start

        return json.dumps({
            "status": "success",
            "url": url,
            "document_id": document_id,
            "chunks": len(chunks),
            "engine": fetch_result.engine,
            "title": extracted.title,
            "elapsed_seconds": round(elapsed, 2),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# Tool: scrape_domain
# ---------------------------------------------------------------------------

@mcp.tool()
def scrape_domain(domain: str, query: str = "", max_results: int = 10, days_back: int = 7) -> str:
    """Search a permitted web domain and return discovered article content.

    Results are also saved under the MCP inbox so the normal pipeline can
    ingest them asynchronously. The domain must be in IPA_SCRAPE_ALLOWED_DOMAINS.
    """
    domain = domain.lower().strip().removeprefix("https://").removeprefix("http://").strip("/")
    valid, reason = _validate_url(f"https://{domain}/")
    if not valid:
        return _json({"status": "error", "domain": domain, "error": reason})
    try:
        from ipa.acquisition.web_scraper import WebScraper, ScrapeSite
        MCP_INBOX.mkdir(parents=True, exist_ok=True)
        site = ScrapeSite(url=f"https://{domain}/", max_articles=max(1, min(max_results, 20)), days_back=max(1, min(days_back, 365)))
        with WebScraper(output_dir=MCP_INBOX, engine="auto", history_db=str(MCP_INBOX / "scrape_history.db")) as scraper:
            summary = scraper.scrape_site(site)
        results = []
        for item in summary.results:
            text = getattr(item, "text", None) or getattr(item, "content", None) or ""
            title = getattr(item, "title", "")
            url = getattr(item, "url", "")
            if query and query.lower() not in f"{title} {text}".lower():
                continue
            results.append({"url": url, "title": title, "text": text[:5000]})
        return _json({"status": "success", "domain": domain, "query": query, "results": results, "saved_to": str(MCP_INBOX)})
    except Exception as exc:
        return _json({"status": "error", "domain": domain, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: ingest_file
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_file(path: str) -> str:
    """Ingest a local file into the knowledge base.

    Supports PDF, HTML, text, and JSON files. Applies safety validation
    (magic bytes, size limits, safe PDF parsing). Chunks with BGE-M3
    semantic chunker, embeds, and indexes in LanceDB + FTS.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        JSON with:
        - status: "success" or "error"
        - document_id: unique ID for the ingested document
        - chunks: number of chunks created
        - pages: number of pages (for PDFs)
        - error: error message if status is "error"
    """
    start = time.monotonic()
    filepath, safety_error = _safe_file(path)

    if safety_error:
        return _json({"status": "error", "path": path, "error": safety_error})

    try:
        # Step 1: Detect file type
        ftype = detect_file_type_from_path(filepath)

        # Step 2: Parse
        from ipa.ingestion.mime_router import detect_mime, route_to_parser
        from ipa.ingestion.parsers import parse

        mime_type = detect_mime(filepath)
        parser_id = route_to_parser(mime_type)

        result = parse(filepath, artifact_id=f"file:{filepath.stem}", parser_id=parser_id)

        if result.status != "parsed" or result.canonical_document is None:
            return json.dumps({
                "status": "error",
                "path": path,
                "error": f"parser {parser_id} failed: {result.status}",
            })

        doc = result.canonical_document

        # Step 3: Semantic chunk
        chunk_fn = _ComponentCache.get_semantic_chunker()
        emb = _ComponentCache.get_embedding_adapter()

        chunks = chunk_fn(doc, embedding_adapter=emb)

        if not chunks:
            return json.dumps({
                "status": "error",
                "path": path,
                "error": "chunking produced no chunks",
            })

        # Step 4: Embed
        chunk_texts = [c.text for c in chunks]
        vectors, sparse_weights = emb.embed_texts_hybrid(chunk_texts)

        # Persist canonical data first, then update the derived vector index.
        store = _ComponentCache.get_document_store()
        store.put_document(doc, artifact_id=f"file:{filepath.stem}")
        store.put_chunks(chunks)
        store.commit()
        index = _ComponentCache.get_lancedb_index()
        index.add_chunks(chunks, vectors, sparse_weights=sparse_weights)
        index.create_fts_index()

        elapsed = time.monotonic() - start

        return json.dumps({
            "status": "success",
            "path": path,
            "document_id": doc.document_id,
            "chunks": len(chunks),
            "pages": doc.pages,
            "mime_type": mime_type,
            "elapsed_seconds": round(elapsed, 2),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "path": path,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# Tool: list_sources
# ---------------------------------------------------------------------------

@mcp.tool()
def list_sources() -> str:
    """List all document sources in the knowledge base.

    Returns distinct sources (URLs, filenames) with chunk counts and
    document counts, useful for understanding what's indexed.

    Returns:
        JSON array of sources, each with:
        - source: document source identifier
        - document_id: document ID
        - mime_type: content type
        - pages: page count
        - chunks: chunk count
    """
    try:
        store = _ComponentCache.get_document_store()
        conn = store._conn  # Access internal connection

        rows = conn.execute(
            "SELECT d.document_id, d.mime_type, d.pages, "
            "COUNT(c.chunk_id) as chunk_count "
            "FROM documents d "
            "LEFT JOIN chunks c ON d.document_id = c.document_id "
            "WHERE d.tombstoned = 0 "
            "GROUP BY d.document_id "
            "ORDER BY chunk_count DESC"
        ).fetchall()

        sources = []
        for doc_id, mime, pages, chunk_count in rows:
            sources.append({
                "document_id": doc_id,
                "mime_type": mime or "unknown",
                "pages": pages,
                "chunks": chunk_count,
            })

        return json.dumps({
            "sources": sources,
            "total_documents": len(sources),
            "total_chunks": sum(s["chunks"] for s in sources),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "sources": []})


# ---------------------------------------------------------------------------
# Tool: get_document
# ---------------------------------------------------------------------------

@mcp.tool()
def get_document(document_id: str) -> str:
    """Retrieve a document's full text and metadata by ID.

    Args:
        document_id: The document ID (from search results or list_sources).

    Returns:
        JSON with:
        - document_id: the document ID
        - text: full document text (may be truncated for very large docs)
        - mime_type: content type
        - pages: page count
        - parser_id: parser used
        - chunks: list of chunk summaries (id, text preview, page)
    """
    try:
        store = _ComponentCache.get_document_store()
        doc = store.get_document(document_id)

        if doc is None:
            return json.dumps({
                "error": f"document not found: {document_id}",
            })

        # Get chunk summaries
        conn = store._conn
        chunk_rows = conn.execute(
            "SELECT chunk_id, substr(text, 1, 100), span_json "
            "FROM chunks WHERE document_id = ? AND tombstoned = 0 "
            "ORDER BY json_extract(metadata_json, '$.chunk_index'), chunk_id",
            (document_id,),
        ).fetchall()

        chunks = []
        for chunk_id, preview, span_json in chunk_rows:
            span = json.loads(span_json) if span_json else {}
            chunks.append({
                "chunk_id": chunk_id,
                "preview": preview + "..." if len(preview) >= 100 else preview,
                "page": span.get("page", 0) if isinstance(span, dict) else 0,
            })

        # Truncate very long text
        text = doc.text
        if len(text) > 10000:
            text = text[:10000] + f"\n\n[... truncated, {len(doc.text)} total chars]"

        return json.dumps({
            "document_id": doc.document_id,
            "text": text,
            "mime_type": doc.mime_type,
            "pages": doc.pages,
            "parser_id": doc.parser_id,
            "chunks": chunks,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_page(chunk: DocumentChunk) -> int:
    """Extract page number from chunk metadata."""
    if chunk.source_span and chunk.source_span.page:
        return chunk.source_span.page
    return 0


def _extract_page_from_id(chunk_id: str) -> int:
    """Try to extract page from chunk_id (best effort)."""
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server via stdio transport."""
    # Ensure LanceDB directory exists
    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)

    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

