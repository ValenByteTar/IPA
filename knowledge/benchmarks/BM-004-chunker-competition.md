---
id: BM-004
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [chunking, ingestion, retrieval, performance]
tags: [fixed-window, recursive, token, semantic, chunks]
related: [BM-003, PAT-002]
supersedes: null
superseded_by: null
---

# BM-004 â€” Competencia de chunkers

## Objetivo

Comparar fixed-window, recursive, token y semantic chunking para el procesamiento reproducible del corpus.

## Entorno

- Dataset: 50 PDFs.
- Artefacto: `outputs/experiments/E5/benchmark_report.json`.
- MÃ©tricas: throughput, duplicaciÃ³n, tamaÃ±o y calidad observable.

## Resultados

- Fixed-window: aproximadamente 48.000 chunks/s, 0% de duplicados y tamaÃ±o uniforme de 512 caracteres.
- Recursive y token: entre 30 y 100 veces mÃ¡s lentos, con mejora marginal observada.
- Semantic chunker: requiere tuning; threshold 0,3, mÃ­nimo 200 y mÃ¡ximo 1200 redujo duplicaciÃ³n a 0,4%.

## ConclusiÃ³n

Fixed-window es el baseline operativo por velocidad, estabilidad e idempotencia. Semantic/adaptive chunking debe activarse selectivamente cuando la calidad adicional justifique el coste.

## Provenance

Artefacto histÃ³rico (`outputs/experiments/E5/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_parser_benchmark.py --mode chunkers`. Relacionado: [PM-002].
