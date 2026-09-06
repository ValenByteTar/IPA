---
id: BM-002
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [vector_index, embeddings, retrieval, performance]
tags: [lancedb, sqlite-vec, bge-m3, vector, indexing]
related: [RES-001]
supersedes: null
superseded_by: null
---

# BM-002 â€” LanceDB vs sqlite-vec

## Objetivo

Comparar dos implementaciones locales de almacenamiento y consulta vectorial bajo el mismo workload de embeddings.

## Entorno

- Corpus: 166k chunks.
- Embeddings: representaciÃ³n vectorial local.
- MÃ©tricas: throughput de indexaciÃ³n, latencia p50 y tamaÃ±o de disco.
- Artefacto: `outputs/experiments/E6-full/benchmark_report.json`.

## Resultados

| MÃ©trica | sqlite-vec | LanceDB |
|---|---:|---:|
| IndexaciÃ³n | 531 chunks/s | 762 chunks/s |
| Latencia p50 | 109 ms | 78 ms |
| Disco | 525 MB | 330 MB |

## ConclusiÃ³n

LanceDB fue preferido para el workload vectorial evaluado por mayor throughput, menor latencia p50 y menor tamaÃ±o de disco.

## Limitaciones

El resultado es especÃ­fico del workload y hardware probado. No convierte a LanceDB en autoridad sobre documentos o metadata canÃ³nica; el Ã­ndice sigue siendo derivado del `DocumentStore`.

## Provenance

Artefacto histÃ³rico (`outputs/experiments/E6-full/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_index_benchmark.py`. Relacionado: [PM-002].
