# Retrieval pipeline

Used by the dashboard chat, the `search_corpus` tool and the Tutor.

```text
query
  -> BGE-M3 dense + sparse (1 forward pass)
  -> LanceDB search_hybrid (dense + FTS tantivy + sparse dot-product, 3-way RRF k=60)
  -> dedup by document
  -> stage-2 rerank (enabled by default; opt-out gate, see below)
  -> top-K into the prompt
```

`BM25Index` (SQLite FTS5) remains the `first_queryable` index: bootstrap and
fallback when there are no vectors. The lexical source inside retrieval is
LanceDB's FTS, not the standalone BM25 index.

## Bulk embedding maintenance (PM-004)

The fast path starts lexical indexing first and drains LanceDB embeddings
separately. For a backlog of **512+ missing vectors**, the drain enters an
exclusive GPU maintenance mode:

- `IPA_EMBED_GPU_MIN_BACKLOG` (default 512) is the threshold for the exclusive
  GPU maintenance lease. Below it, chat remains available and there is no bulk
  lease; the adapter still resolves `device=auto` through the VRAM gate and can
  use CUDA if enough physical memory is free. Pin CPU with `IPA_EMBED_DEVICE=cpu`;
- GPU bulk uses BGE-M3 FP16 with batch 4 (both CPU/GPU batch defaults are 4;
  `IPA_EMBED_BATCH_CPU` / `IPA_EMBED_BATCH_GPU` can tune per machine);
- the worker publishes `outputs/web_dashboard/embedding_maintenance.json`,
  claims `outputs/agent/vram.lock`, waits for active Ollama generation to finish,
  unloads Ollama, and holds BGE on CUDA through the **entire** backlog;
- chat requests are rejected with HTTP 423 and the chat panel shows progress.
  The idle scheduler (T1/T2) and manual promotion/review/reindex routes are
  gated while the lease is active, preventing promotion/purge from mutating a
  source corpus during embedding. The worker restores the Ollama model before
  clearing the maintenance state;
- partial LanceDB writes are the checkpoint. On restart, existing `chunk_id`s
  are skipped and only the remainder is embedded. A dead worker PID changes the
  dashboard state to `interrupted` and re-enables chat; rerun the drain to resume;
- `IPA_EMBED_GPU_BULK=0` / `run_embed_drain.py --cpu-only` disable the exclusive
  GPU bulk lease; they do **not** pin the model device. With `device=auto`, use
  `IPA_EMBED_DEVICE=cpu` to guarantee CPU. `IPA_EMBED_GPU_WAIT_SECONDS` bounds
  waiting for an existing VRAM owner; if BGE cannot load safely, the worker
  restores chat and continues on CPU.

The user-visible chat outage is intentional for a large bulk job; it is never
silent. The verified microbenchmark on the same 64 real chunks found CPU FP32
batch 4 at 2.86–2.99 chunks/s vs GPU FP16 batch 4/8 at 124–135 chunks/s.
This is a small-sample device-tuning baseline, not a full-corpus SLA; the
threshold is conservative relative to the measured ~240-chunk worst-case
switching break-even. Details: `knowledge/postmortems/PM-004`.

## Precision tuning status

- Standard `EmbeddingAdapter` defaults to FP32 + batch 4 on CPU and FP16 + batch
  4 on CUDA. The 6-thread CPU result was measured with that thread count set by
  the probe; production code does not pin PyTorch threads. Explicit caller batch
  overrides remain effective (for example, the separate continuous pipeline).
- `IPA_RERANK_DEVICE=auto` may select CUDA when the physical-free-VRAM gate passes;
  it is not a CPU-only default. CPU uses FP32. The current wrapper inherits
  FlagReranker's batch default and passes `max_length=8192`; CPU batch/length
  tuning has not been evaluated separately.
- FP8 E4M3/NVIDIA scaling on Ada is only a **proposed experiment** for BGE-M3.
  The current FlagEmbedding adapter has no FP8 mode; see
  `knowledge/experiments/EXP-009-bge-m3-fp8-rtx4050.md`. No FP8 code path,
  benchmark, or production setting has been added.

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
