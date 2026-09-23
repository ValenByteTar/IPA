"""EmbeddingAdapter â€” generate vector embeddings for chunks.

Uses FlagEmbedding's BGEM3FlagModel (BAAI/bge-m3) by default.
BGE-M3: 1024 dims, 8192 max tokens, multilingual (100+ languages),
supports dense + sparse + multi-vector retrieval. MTEB 63.0.

Key features:
  - FP16 inference on GPU (2x speedup, half VRAM)
  - Dense embeddings (1024 dims) for semantic similarity
  - Sparse embeddings (learned token weights, BM25-like) for keyword matching
  - Both produced in a single forward pass (sparse is "free")
  - Optional query instruction prefix for asymmetric retrieval
  - Proper VRAM release on close() (gc.collect + empty_cache)

The model is downloaded once and cached locally.  No API calls, no network
dependency after the initial download.

This adapter is shared by all vector indexes (LanceDB, sqlite-vec, etc.).
It separates embedding cost from storage/search cost, as required by E7.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ipa.contracts import DocumentChunk


# Default model: BGE-M3 â€” 1024 dims, 8K context, multilingual, MTEB 63.0.
# Previous: all-MiniLM-L6-v2 (384 dims, 256 tokens, MTEB 56.3) â€” truncated
# chunks to 256 tokens and had weaker retrieval quality.
DEFAULT_MODEL = "BAAI/bge-m3"
# Batch por device (None en el constructor = auto). El probe inicial de GPU
# usó textos sintéticos cortos y favoreció batch 64; con chunks reales el
# throughput máximo observado fue batch 4 en ambos dispositivos. El probe CPU
# fijó torch a 6 threads (~2.9 chunks/s); el adapter no fija ese valor. GPU
# fp16 ≈129 chunks/s (PM-004).
DEFAULT_BATCH_GPU = int(os.environ.get("IPA_EMBED_BATCH_GPU", "4") or 4)
DEFAULT_BATCH_CPU = int(os.environ.get("IPA_EMBED_BATCH_CPU", "4") or 4)
DEFAULT_DIM = 1024


class _QueryEmbeddingCache:
    """LRU de embeddings de query (query → dense/sparse).

    Las queries repetidas (misma pregunta, retrieval de turnos seguidos,
    herramientas que re-consultan) pagaban BGE-M3 completo cada vez:
    ~100-300ms + GPU por repetida. Desactivar con IPA_EMBED_CACHE_SIZE=0.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

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


class EmbeddingAdapter:
    """Generate embeddings for text chunks using FlagEmbedding's BGE-M3.

    Produces dense (1024-dim) and sparse (token-weight dict) embeddings
    in a single forward pass.  Uses FP16 on GPU for 2x speedup.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int | None = None,
        show_progress: bool = True,
        device: str = "auto",
        use_fp16: bool = True,
        query_instruction: str | None = None,
        max_length: int = 2048,
    ) -> None:
        self.model_name = model_name
        # None = auto por device; `batch_size` explícito mantiene prioridad.
        # El batch 64 del probe histórico era un tuning con texto sintético;
        # con corpus real variable, batch 4 rindió mejor en ambos devices
        # (PM-004, microbenchmark 2026-09-22).
        self._batch_override = batch_size
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_GPU
        self.show_progress = show_progress
        # IPA_EMBED_DEVICE permite forzar el device cuando el caller no lo
        # especifica (device="auto"): los tests lo setean a "cpu" para ser
        # herméticos — cargar BGE-M3 en una GPU ya ocupada por el dashboard
        # agota la VRAM y congela la UI (el compositor de Windows se queda
        # sin memoria de video).
        if device == "auto":
            device = os.environ.get("IPA_EMBED_DEVICE", device) or device
        self.device = device
        self.use_fp16 = use_fp16
        # BGE-M3 supports query instruction prefixes for asymmetric retrieval
        # (e.g. "Represent this sentence for searching relevant passages: ").
        # When set, queries are prefixed before encoding; passages are not.
        self.query_instruction = query_instruction
        self.max_length = max_length
        self._model = None
        self._dim = None
        self._device_resolved = None
        # Cache de embeddings de query: queries repetidas no re-corren BGE-M3.
        # IPA_EMBED_CACHE_SIZE=0 lo desactiva.
        _size = os.environ.get("IPA_EMBED_CACHE_SIZE", "").strip()
        self._query_cache = _QueryEmbeddingCache(
            maxsize=int(_size) if _size.isdigit() else 256)

    def _resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' solo con headroom de VRAM, si no 'cpu'.

        Gate igual al del reranker: el LLM del chat es dueño de la GPU.
        Cargar BGE-M3 (~2.2 GB) en una GPU casi llena agotó la VRAM y congeló
        la UI completa (el compositor de Windows se quedó sin memoria de
        video) — bug real medido en EXP-008 §10. En Windows/WDDM
        torch.cuda.mem_get_info() sobreestima la libre: usar nvidia-smi.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                from ipa.indexes.reranker_adapter import physical_free_vram_mb
                min_free_mb = float(os.environ.get("IPA_EMBED_MIN_FREE_MB", "2048"))
                free_mb = physical_free_vram_mb()
                if free_mb is None:
                    free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                if free_mb >= min_free_mb:
                    return "cuda"
                print(
                    f"[embed] VRAM libre {free_mb:.0f} MiB < {min_free_mb:.0f} "
                    "— BGE-M3 en CPU (gate de VRAM)", flush=True,
                )
        except ImportError:
            pass
        return "cpu"

    def _resolve_batch(self) -> int:
        """Batch efectivo por device (override explícito gana)."""
        if self._batch_override is not None:
            return int(self._batch_override)
        return DEFAULT_BATCH_GPU if self._device_resolved == "cuda" else DEFAULT_BATCH_CPU

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel
        self._device_resolved = self._resolve_device()
        self.batch_size = self._resolve_batch()
        # FP16 only helps on GPU; on CPU it's slower.
        fp16 = self.use_fp16 and self._device_resolved == "cuda"
        # Skip the hub round-trip ("Fetching N files") when the snapshot is
        # already in the local HF cache — pero HF_HUB_OFFLINE se acota a este
        # load: dejarlo seteado fuga al resto del proceso y rompe otros
        # consumidores de HF que sí necesitan red (bug real en CI: docling
        # no podía bajar sus modelos tras un test que cargaba BGE-M3).
        cached = False
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = isinstance(
                try_to_load_from_cache(self.model_name, "config.json"), str)
        except Exception:
            pass
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if cached and prev_offline is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            # BUG FIX: BGEM3FlagModel (M3Embedder) accepts `devices` (plural),
            # not `device`. The singular `device` was swallowed by **kwargs and
            # silently ignored, so device="cpu" still ran on cuda:0.
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=fp16,
                devices=[self._device_resolved] if self._device_resolved else None,
            )
        finally:
            if cached and prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
        self._dim = DEFAULT_DIM

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dim is None:
            self._ensure_model()
        return self._dim  # type: ignore

    @property
    def active_device(self) -> str:
        """Return the resolved device ('cuda' or 'cpu')."""
        if self._device_resolved is None:
            self._ensure_model()
        return self._device_resolved  # type: ignore

    def try_move_to_gpu(self, min_free_mb: float | None = None) -> bool:
        """Sube el modelo a GPU si hay headroom (venía resuelto en CPU).

        Medido 2026-09-22 (PM-004): con los mismos 64 chunks reales (~512 chars),
        mejor CPU FP32 batch 4, con 6 threads fijados en la sonda, dio
        2.86-2.99 chunks/s; GPU FP16 batch 4 dio 124-135 chunks/s (~45x). GPU
        FP32 batch 8 dio ~37 chunks/s. El adapter no fija el thread count. El
        probe anterior (~125x) mezclaba texto sintético corto y chunks reales.
        El locking de VRAM queda en el caller: en 6 GB BGE-M3 y el LLM no conviven
        (OOM medido). Devuelve True si quedó GPU.
        """
        if self._device_resolved == "cuda":
            return True
        try:
            from ipa.indexes.reranker_adapter import physical_free_vram_mb
            free_mb = physical_free_vram_mb()
        except Exception:
            free_mb = None
        threshold = float(min_free_mb if min_free_mb is not None
                          else os.environ.get("IPA_EMBED_MIN_FREE_MB", "2048"))
        if free_mb is None or free_mb < threshold:
            return False
        self.close()
        self.device = "cuda"
        self._device_resolved = None
        try:
            self._ensure_model()
        except Exception:
            # Carga en GPU falló (OOM u otro): volver a CPU sin romper el drain.
            self.device = "cpu"
            self._device_resolved = None
            self._ensure_model()
            return False
        return self._device_resolved == "cuda"

    def move_to_cpu(self) -> None:
        """Baja el modelo a CPU y libera la VRAM (cierra una ventana GPU)."""
        if self._device_resolved != "cuda":
            return
        self.close()
        self.device = "cpu"
        self._device_resolved = None
        self._ensure_model()

    def release_gpu(self) -> None:
        """Libera la ventana GPU sin recargar el modelo (lazy).

        close() solo suelta la VRAM pero deja device/_device_resolved en
        "cuda": un embed posterior recargaría BGE-M3 en CUDA encima del chat
        recién restaurado (OOM medido en 6 GB). Acá el próximo embed
        re-resuelve el device — normalmente CPU por el gate de VRAM.
        """
        if self._device_resolved != "cuda" and self.device != "cuda":
            return
        self.close()
        self.device = "cpu"
        self._device_resolved = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.  Returns a list of float vectors (dense only)."""
        result = self._encode(texts, return_dense=True, return_sparse=False)
        return [v.tolist() for v in result["dense_vecs"]]

    def embed_texts_hybrid(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """Embed texts with both dense and sparse representations.

        Returns (dense_vectors, sparse_weights) where:
          - dense_vectors: list of 1024-dim float lists
          - sparse_weights: list of dicts {token_id: weight}
        Both are produced in a single forward pass (sparse is free).
        """
        result = self._encode(texts, return_dense=True, return_sparse=True)
        dense = [v.tolist() for v in result["dense_vecs"]]
        sparse = result["lexical_weights"]
        return dense, sparse

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[tuple[str, list[float]]]:
        """Embed a list of chunks.  Returns (chunk_id, vector) pairs."""
        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)
        return list(zip([c.chunk_id for c in chunks], vectors))

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (dense only).

        If query_instruction is set, the instruction is prepended to the
        query before encoding (asymmetric retrieval improvement).
        """
        cached = self._query_cache.get(query)
        if cached is not None:
            return list(cached[0])
        text = self._apply_query_instruction(query)
        result = self._encode(
            [text], return_dense=True, return_sparse=False,
            show_progress=False,
        )
        dense = result["dense_vecs"][0].tolist()
        self._query_cache.put(query, (dense, None))
        return list(dense)

    def embed_query_hybrid(
        self,
        query: str,
    ) -> tuple[list[float], dict[int, float]]:
        """Embed a query with both dense and sparse representations.

        Returns (dense_vector, sparse_weights).
        If query_instruction is set, it is prepended to the query.
        """
        cached = self._query_cache.get(query)
        if cached is not None and cached[1] is not None:
            return list(cached[0]), dict(cached[1])
        text = self._apply_query_instruction(query)
        result = self._encode(
            [text], return_dense=True, return_sparse=True,
            show_progress=False,
        )
        dense = result["dense_vecs"][0].tolist()
        sparse = result["lexical_weights"][0]
        self._query_cache.put(query, (dense, sparse))
        return list(dense), dict(sparse)

    def query_cache_stats(self) -> dict[str, int]:
        """Hits/misses del cache de queries (observabilidad)."""
        return self._query_cache.stats()

    def _apply_query_instruction(self, query: str) -> str:
        """Prepend query instruction if configured."""
        if self.query_instruction:
            return self.query_instruction + query
        return query

    def _encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False,
        show_progress: bool | None = None,
    ) -> dict[str, Any]:
        """Internal encode wrapper that handles batching and progress."""
        self._ensure_model()
        if show_progress is None:
            show_progress = self.show_progress
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

    def close(self) -> None:
        """Release model resources and free GPU VRAM."""
        if self._model is not None:
            self._model = None
            self._dim = None
            # Force VRAM release on GPU â€” without this, VRAM stays
            # allocated until Python GC runs (which may be never
            # during a long-running process).
            try:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def __enter__(self) -> "EmbeddingAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

