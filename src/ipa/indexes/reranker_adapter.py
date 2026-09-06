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

from dataclasses import dataclass, field
from typing import Any

from ipa.contracts import SearchHit


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


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
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker
        self._device_resolved = self._resolve_device()
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

