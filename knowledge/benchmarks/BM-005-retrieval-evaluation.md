---
id: BM-005
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [retrieval, lexical_index, vector_index, evaluation]
tags: [tantivy, lancedb, hybrid, recall, mrr, rrf]
related: [BM-001, BM-002]
supersedes: null
superseded_by: null
evidence: ["scripts/benchmarks/run_retrieval_eval.py", "tests/test_retrieval_eval.py"]
affects: ["scripts/benchmarks/run_retrieval_eval.py"]
---

# BM-005 â€” Evaluación lexical, vectorial e híbrida

## Objetivo

Medir qué backend funciona mejor según la forma de la consulta y evaluar si la fusión híbrida conserva cobertura.

## Entorno

- Dataset: 200 consultas sobre 166k chunks.
- Artefacto: `outputs/experiments/E10/benchmark_report.json` (histórico, ya no en disco — ver PM-002).
- Métricas: recall@10, MRR y latencia p50.

## Resultados

- Tantivy en consultas derivadas de términos: recall@10 98,5%, MRR 0,91 y p50 0 ms.
- LanceDB en consultas cortas: recall@10 58%, MRR 0,37 y p50 78 ms.
- La fusión híbrida recupera cobertura, aunque puede diluir el MRR lexical.

## Conclusión

Tantivy es la opción fuerte para términos exactos, identificadores y nombres propios. Hybrid retrieval es preferible para lenguaje natural donde la coincidencia lexical puede fallar.

## Limitaciones

El conjunto de consultas no representa toda interacción futura. Las métricas deben complementarse con evaluación de utilidad, evidencia y citas.

## Provenance

Artefacto histórico (`outputs/experiments/E10/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runner reproducible en `scripts/benchmarks/run_retrieval_eval.py`. Relacionado: [PM-002].
