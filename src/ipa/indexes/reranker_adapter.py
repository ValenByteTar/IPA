"""RerankerAdapter â€” cross-encoder reranking for retrieval results.

Uses BGE-reranker-v2-m3 (BAAI/bge-reranker-v2-m3), a cross-encoder that
reads query + document together and produces a relevance score.

Two-stage retrieval pipeline:
  1. BGE-M3 dense/sparse retrieval â†’ top-N candidates (fast, approximate)
  2. BGE-reranker cross-encoder â†’ rerank top-N â†’ top-K (slow, precise)

The reranker is a 568M parameter XLM-RoBERTa model that processes query-doc
pairs jointly (not separately like embedding models). This late interaction
captures fine-grained relevance that cosine similarity misses.

Model: BAAI/bge-reranker-v2-m3
  - 568M params, ~2.1 GB
  - Multilingual (100+ languages)
  - Max input: 8192 tokens
  - FP16 on GPU for 2x speedup
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ipa.contracts import SearchHit


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Env gates (rerank ON by default — measured in E10-rerank: +20.5pp recall@1
# on lancedb_hybrid for ~+0.65s/query GPU, ~+0.7s/query on the CPU fallback):
#   IPA_RERANK=0            opt-out — disables reranking in the retrieval paths
#   IPA_RERANK_DEVICE       auto|cuda|cpu (default auto)
#   IPA_RERANK_MIN_FREE_MB  min free VRAM to run on GPU (default 2048) else CPU
#   IPA_RERANK_CACHE_SIZE   entradas del cache query→ranking (default 128, 0 off)


class _RerankCache:
    """LRU de rankings (query + set de candidatos → orden).

    El cross-encoder re-scorea los MISMOS chunks en queries repetidas o muy
    similares (turnos seguidos del chat re-consultan lo mismo). La clave
    incluye el hash del texto de los candidatos, así un corpus distinto no
    reusa un ranking viejo.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(query: str, texts: list[str], top_k: int) -> str:
        h = hashlib.sha256()
        h.update(query.encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update(str(top_k).encode())
        for t in texts:
            h.update(b"\x01")
            h.update(t[:400].encode("utf-8", "ignore"))
        return h.hexdigest()

    def get(self, key: str) -> Any | None:
        if self.maxsize <= 0:
            return None
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._data), "maxsize": self.maxsize,
                "hits": self.hits, "misses": self.misses}


_rerank_cache = _RerankCache(
    maxsize=int(os.environ.get("IPA_RERANK_CACHE_SIZE", "128") or 128))


def rerank_cache_stats() -> dict[str, int]:
    """Hits/misses del cache de reranking (observabilidad)."""
    return _rerank_cache.stats()


def rerank_enabled() -> bool:
    return os.environ.get("IPA_RERANK", "1").strip().lower() not in (
        "0", "false", "no", "off")


def physical_free_vram_mb() -> float | None:
    """VRAM libre física según el driver (nvidia-smi).

    torch.cuda.mem_get_info() sobreestima en Windows/WDDM (cuenta la memoria
    compartida del sistema como libre): con el LLM ocupando ~4.5 GB reporta
    ~5 GB libres y el gate mandaría el reranker a GPU igual. None si
    nvidia-smi no está disponible (el caller cae a mem_get_info).
    """
    try:
        import subprocess
        no_window = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                     if os.name == "nt" else 0)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=no_window,
        )
        used, total = (
            float(x) for x in out.stdout.strip().splitlines()[0].split(",")
        )
        return total - used
    except Exception:
        return None


_SHARED_RERANKER: "RerankerAdapter | None" = None


def get_shared_reranker() -> "RerankerAdapter":
    """Process-wide lazy singleton — the model is ~2.1 GB, load it once."""
    global _SHARED_RERANKER
    if _SHARED_RERANKER is None:
        _SHARED_RERANKER = RerankerAdapter(
            device=os.environ.get("IPA_RERANK_DEVICE", "auto")
        )
    return _SHARED_RERANKER


def maybe_rerank(
    query: str,
    items: list[dict[str, Any]],
    top_k: int,
    *,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """Rerank a list of hit dicts unless IPA_RERANK=0; passthrough otherwise.

    Each item needs a text field (text_key) and keeps its original payload —
    only the order and the score change (score becomes the cross-encoder
    score, and 'reranked': True marks the ordering provenance).
    """
    if not rerank_enabled() or not items:
        return items[:top_k]
    try:
        texts = [str(it.get(text_key) or "") for it in items]
        ckey = _rerank_cache.key(query, texts, top_k)
        cached = _rerank_cache.get(ckey)
        if cached is not None:
            return [
                {**items[idx], "score": score, "reranked": True}
                for idx, score in cached
            ]
        cands = [
            RerankCandidate(
                chunk_id=str(i),
                text=str(it.get(text_key) or ""),
                score=float(it.get("score") or 0.0),
            )
            for i, it in enumerate(items)
        ]
        ranked = get_shared_reranker().rerank(query, cands, top_k=top_k)
        _rerank_cache.put(ckey, [(int(c.chunk_id), round(c.score, 4)) for c in ranked])
        return [
            {**items[int(c.chunk_id)], "score": round(c.score, 4), "reranked": True}
            for c in ranked
        ]
    except Exception:
        return items[:top_k]


@dataclass(frozen=True)
class RerankCandidate:
    """A retrieval candidate with text, for reranking.

    Wraps a SearchHit with the chunk text needed by the cross-encoder.
    """
    chunk_id: str
    text: str
    score: float
    source_span: Any = None
    retrieval_backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hit(
        cls,
        hit: SearchHit,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> "RerankCandidate":
        return cls(
            chunk_id=hit.chunk_id,
            text=text,
            score=hit.score,
            source_span=hit.source_span,
            retrieval_backend=hit.retrieval_backend,
            metadata=metadata or {},
        )


class RerankerAdapter:
    """Cross-encoder reranker using BGE-reranker-v2-m3.

    Reranks retrieval candidates by reading query + document jointly.
    Uses FP16 on GPU for 2x speedup.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        use_fp16: bool = True,
        device: str = "auto",
        max_length: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.max_length = max_length
        self._model = None
        self._device_resolved = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                # VRAM headroom gate: the chat LLM owns the GPU — run the
                # reranker on CPU when there isn't room, rather than
                # contending for VRAM mid-conversation. Usa la VRAM física
                # (nvidia-smi): mem_get_info sobreestima en WDDM y el gate
                # mandaría el reranker a GPU con el LLM cargado.
                min_free_mb = float(os.environ.get("IPA_RERANK_MIN_FREE_MB", "2048"))
                free_mb = physical_free_vram_mb()
                if free_mb is None:
                    free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                if free_mb >= min_free_mb:
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker
        self._device_resolved = self._resolve_device()
        print(f"[rerank] cross-encoder loading on {self._device_resolved}", flush=True)
        fp16 = self.use_fp16 and self._device_resolved == "cuda"
        self._model = FlagReranker(
            self.model_name,
            use_fp16=fp16,
            device=self._device_resolved,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int = 5,
        normalize: bool = True,
    ) -> list[RerankCandidate]:
        """Rerank retrieval candidates by cross-encoder relevance scoring.

        Args:
            query: The search query string.
            candidates: List of RerankCandidate with chunk text.
            top_k: Number of top results to return.
            normalize: If True, normalize scores to [0, 1] via sigmoid.

        Returns:
            Top-k RerankCandidate sorted by reranker score (descending).
            The score field is updated with the reranker score.
        """
        if not candidates:
            return []
        self._ensure_model()

        # Build query-doc pairs
        pairs = [[query, c.text] for c in candidates]

        # Score all pairs
        scores = self._model.compute_score(
            pairs,
            max_length=self.max_length,
            normalize=normalize,
        )

        # Handle single-element case (compute_score returns float, not list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Sort by reranker score (descending)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with updated scores
        result = []
        for cand, score in scored[:top_k]:
            result.append(RerankCandidate(
                chunk_id=cand.chunk_id,
                text=cand.text,
                score=float(score),
                source_span=cand.source_span,
                retrieval_backend=cand.retrieval_backend,
                metadata=cand.metadata,
            ))
        return result

    def close(self) -> None:
        if self._model is not None:
            self._model = None

    def __enter__(self) -> "RerankerAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

