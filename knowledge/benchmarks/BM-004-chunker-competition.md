---
id: BM-004
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [chunking, ingestion, retrieval, performance]
tags: [fixed-window, recursive, token, semantic, chunks]
related: [BM-003, PAT-002]
supersedes: null
superseded_by: null
evidence: ["scripts/benchmarks/run_parser_benchmark.py", "tests/test_chunkers_alt.py"]
affects: ["scripts/benchmarks/run_parser_benchmark.py"]
---

# BM-004 â€” Competencia de chunkers

## Objetivo

Comparar fixed-window, recursive, token y semantic chunking para el procesamiento reproducible del corpus.

## Entorno

- Dataset: 50 PDFs.
- Artefacto: `outputs/experiments/E5/benchmark_report.json` (histórico, ya no en disco — ver PM-002).
- Métricas: throughput, duplicación, tamaño y calidad observable.

## Resultados

- Fixed-window: aproximadamente 48.000 chunks/s, 0% de duplicados y tamaño uniforme de 512 caracteres.
- Recursive y token: entre 30 y 100 veces más lentos, con mejora marginal observada.
- Semantic chunker: requiere tuning; threshold 0,3, mínimo 200 y máximo 1200 redujo duplicación a 0,4%.

## Conclusión

Fixed-window es el baseline operativo por velocidad, estabilidad e idempotencia. Semantic/adaptive chunking debe activarse selectivamente cuando la calidad adicional justifique el coste.

## Provenance

Artefacto histórico (`outputs/experiments/E5/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_parser_benchmark.py --mode chunkers`. Relacionado: [PM-002].
