---
id: BM-005
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [retrieval, lexical_index, vector_index, evaluation]
tags: [tantivy, lancedb, hybrid, recall, mrr, rrf]
related: [BM-001, BM-002]
supersedes: null
superseded_by: null
---

# BM-005 â€” EvaluaciÃ³n lexical, vectorial e hÃ­brida

## Objetivo

Medir quÃ© backend funciona mejor segÃºn la forma de la consulta y evaluar si la fusiÃ³n hÃ­brida conserva cobertura.

## Entorno

- Dataset: 200 consultas sobre 166k chunks.
- Artefacto: `outputs/experiments/E10/benchmark_report.json`.
- MÃ©tricas: recall@10, MRR y latencia p50.

## Resultados

- Tantivy en consultas derivadas de tÃ©rminos: recall@10 98,5%, MRR 0,91 y p50 0 ms.
- LanceDB en consultas cortas: recall@10 58%, MRR 0,37 y p50 78 ms.
- La fusiÃ³n hÃ­brida recupera cobertura, aunque puede diluir el MRR lexical.

## ConclusiÃ³n

Tantivy es la opciÃ³n fuerte para tÃ©rminos exactos, identificadores y nombres propios. Hybrid retrieval es preferible para lenguaje natural donde la coincidencia lexical puede fallar.

## Limitaciones

El conjunto de consultas no representa toda interacciÃ³n futura. Las mÃ©tricas deben complementarse con evaluaciÃ³n de utilidad, evidencia y citas.

## Provenance

Artefacto histÃ³rico (`outputs/experiments/E10/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_retrieval_eval.py`. Relacionado: [PM-002].
