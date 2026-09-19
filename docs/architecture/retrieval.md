# Retrieval pipeline

Used by the dashboard chat, the `search_corpus` tool and the Tutor.

```text
query
  -> BGE-M3 dense + sparse (1 forward pass)
  -> LanceDB search_hybrid (dense + FTS tantivy + sparse dot-product, 3-way RRF k=60)
  -> dedup by document
  -> stage-2 rerank (opt-in gate, see below)
  -> top-K into the prompt
```

`BM25Index` (SQLite FTS5) remains the `first_queryable` index: bootstrap and
fallback when there are no vectors. The lexical source inside retrieval is
LanceDB's FTS, not the standalone BM25 index.

## LanceDB schema (derived, rebuildable)

Scalar metadata columns for pre-filtering with `.where()`:

- `source_domain`, `published_at`, `provenance`, `quality_score`;
- `published_at` is a proxy for `documents.stored_at` — the canonical store
  does not track a publication date.

Sync paths:

- `add_chunks(doc_meta=...)` on ingestion;
- `sync_doc_metadata()` on promotion merge and warmup;
- `scripts/operations/sync_index_metadata.py` for manual backfill
  (`--all` forces a full resync).

## Stage-2 rerank (cross-encoder)

`BAAI/bge-reranker-v2-m3` (~2.1 GB) reorders the top candidates. It competes
with the 9B LLM for VRAM, so it is gated:

- `IPA_RERANK` — default ON; opt-out with `0`/`false`/`no`/`off`;
- `IPA_RERANK_DEVICE` — `auto` (default) | `cuda` | `cpu`;
- `IPA_RERANK_MIN_FREE_MB` — default 2048; below it the reranker runs on CPU.

The gate measures **physical** free VRAM via `nvidia-smi`
(`reranker_adapter.physical_free_vram_mb()`). `torch.cuda.mem_get_info()`
overreports on Windows/WDDM (it counts shared system memory as free): with the
LLM holding ~4.5 GB it reported ~5 GB free and the reranker loaded on GPU
anyway. Falls back to `mem_get_info` when `nvidia-smi` is unavailable.

`reranker_adapter.maybe_rerank()` is a process-wide lazy singleton with
passthrough fallback on failure. The eval runner honours
`IPA_RERANK_DEVICE`; the MCP server's `search_knowledge` reranks
unconditionally through its own cache.

Measured impact (E10-rerank, 200 queries, `lancedb_hybrid`, corpus E12):
recall@1 0.465 → 0.670 (+20.5pp), MRR +0.141, nDCG@10 +0.117, recall@20
unchanged (it reorders, it does not expand); ~+0.65 s/query on GPU, ~+0.7 s on
the CPU fallback. Evidence: `outputs/experiments/E10-rerank/` and
`knowledge/experiments/EXP-007-rerank-stage2.md`.
