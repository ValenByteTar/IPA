---
id: EXP-007
category: experiment
status: accepted
created: 2026-09-18
updated: 2026-09-18
author: agent
components: [retrieval, indexes, reranker]
tags: [rerank, cross-encoder, hybrid-retrieval, vram-gate, wddm, evaluation]
related: [EXP-003, PAT-004, DEC-006]
supersedes: null
superseded_by: null
---

# EXP-007 — Rerank stage-2 (cross-encoder) sobre el retrieval híbrido

## Hipótesis

Un cross-encoder stage-2 (BGE-reranker-v2-m3) sobre el backend de producción
(`lancedb_hybrid`, 3-way RRF) mejora la precisión de ordenamiento lo suficiente
como para justificar su costo de VRAM/latencia, con el gate mandándolo a CPU
cuando el LLM ocupa la GPU.

## Configuración

- **Corpus**: `outputs/experiments/E12-corpus` — 139,102 chunks.
- **Backend**: `lancedb_hybrid` (dense + FTS tantivy + sparse dot-product, RRF k=60).
- **Query set**: 200 queries sintéticas, seed 42, k=20 (idéntico entre corridas).
- **Reranker**: `BAAI/bge-reranker-v2-m3`, fp16 en CUDA / fp32 en CPU.
- **Runner**: `scripts/benchmarks/run_retrieval_eval.py --rerank`
  (`--where` para pre-filtros de metadata).
- **Hardware**: RTX 4050 6 GB. GPU: BGE-M3 en CPU (dashboard), LLM en VRAM.

## Resultados

### GPU — 200 queries

| Métrica | Baseline | Con rerank | Δ |
|---|---|---|---|
| recall@1 | 0.4650 | 0.6700 | **+20.5pp** |
| recall@5 | 0.7150 | 0.7900 | +7.5pp |
| recall@10 | 0.7600 | 0.7950 | +3.5pp |
| recall@20 | 0.8050 | 0.8050 | 0 |
| MRR | 0.5826 | 0.7239 | +0.141 |
| nDCG@10 | 0.6243 | 0.7414 | +0.117 |
| latencia avg | 2015 ms | 2664 ms | +649 ms |

### CPU (fallback real, `CUDA_VISIBLE_DEVICES=""`) — 30 queries

Misma ganancia de calidad (recall@1 +20.0pp, MRR +0.144) a ~+0.7 s/query.

### Verificación en vivo (dashboard)

Con el LLM en VRAM (4,486 MiB usados de 6,141) y rerank forzado a CPU: fase de
retrieval **3.2 s** en régimen (embed BGE-M3 CPU + search_hybrid + rerank CPU),
lección end-to-end ~20 s. Sin errores ni OOM.

## Conclusiones

- El cross-encoder **justifica su costo**: +20.5pp recall@1 es la mayor ganancia
  de una sola etapa medida en el proyecto (E9 enrichment: +14.3% recall@10).
- recall@20 no cambia: reordena los mismos candidatos, no expande el pool — la
  ganancia es precisión en el top-K, que es lo que va al prompt.
- El costo de latencia (~0.65 s GPU / ~0.7 s CPU) es aceptable en ambos modos.
- Default: **ON** (opt-out `IPA_RERANK=0`).

## Hallazgo de robustez (gate de VRAM)

El gate usaba `torch.cuda.mem_get_info()`, que en Windows/WDDM **sobreestima la
VRAM libre** (cuenta memoria compartida): con el LLM ocupando 4.5 GB reportó
~5,080 MB libres cuando la física libre era 1,655 MB, y el reranker cargaba en
GPU de todos modos. Fix: medir la VRAM física con `nvidia-smi`
(`physical_free_vram_mb()`) y caer a `mem_get_info` solo si no está disponible.
Verificado en vivo: con el LLM cargado → CPU; sin LLM → CUDA.

## Evidencia

- `outputs/experiments/E10-rerank/` (`baseline/`, `rerank/`, `baseline-cpu/`,
  `rerank-cpu2/`, `comparison.json`).
- `docs/architecture/retrieval.md`.
- Tests: `tests/test_index_adapters.py` (gate físico + passthrough),
  `tests/test_mcp_server.py` (rerank aplicado en el MCP).
