---
id: BM-002
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [vector_index, embeddings, retrieval, performance]
tags: [lancedb, sqlite-vec, bge-m3, vector, indexing]
related: [RES-001]
supersedes: null
superseded_by: null
evidence: ["scripts/benchmarks/run_index_benchmark.py", "tests/test_index_adapters.py"]
affects: ["scripts/benchmarks/run_index_benchmark.py", "src/ipa/indexes/**"]
---

# BM-002 â€” LanceDB vs sqlite-vec

## Objetivo

Comparar dos implementaciones locales de almacenamiento y consulta vectorial bajo el mismo workload de embeddings.

## Entorno

- Corpus: 166k chunks.
- Embeddings: representación vectorial local.
- Métricas: throughput de indexación, latencia p50 y tamaño de disco.
- Artefacto: `outputs/experiments/E6-full/benchmark_report.json` (histórico, ya no en disco — ver PM-002).

## Resultados

| Métrica | sqlite-vec | LanceDB |
|---|---:|---:|
| Indexación | 531 chunks/s | 762 chunks/s |
| Latencia p50 | 109 ms | 78 ms |
| Disco | 525 MB | 330 MB |

## Conclusión

LanceDB fue preferido para el workload vectorial evaluado por mayor throughput, menor latencia p50 y menor tamaño de disco.

## Limitaciones

El resultado es específico del workload y hardware probado. No convierte a LanceDB en autoridad sobre documentos o metadata canónica; el índice sigue siendo derivado del `DocumentStore`.

## Provenance

Artefacto histórico (`outputs/experiments/E6-full/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_index_benchmark.py`. Relacionado: [PM-002].
