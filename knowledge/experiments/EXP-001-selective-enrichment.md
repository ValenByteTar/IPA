---
id: EXP-001
category: experiment
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [enrichment, retrieval, lexical_index, vector_index]
tags: [summary, synthetic_queries, claims, qwen, recall]
related: [BM-005, PAT-003]
supersedes: null
superseded_by: null
---

# EXP-001 â€” Enrichment selectivo para retrieval

## HipÃ³tesis

Summaries, synthetic queries y claims generados selectivamente pueden mejorar retrieval sin reemplazar el texto canÃ³nico.

## ConfiguraciÃ³n

- Dataset: 2.000 chunks.
- Modelo: qwen3.5:4b-q4_K_M, `think=False`, 3 workers.
- Total: 8.000 inferencias.
- Artefacto: `outputs/experiments/E9-experiment/benchmark_report.json`.

## Resultados

- Las tres estrategias mejoraron retrieval en consultas naturales.
- Summary fue la mejor estrategia para Tantivy: +14,3% recall@10.
- Synthetic queries fue la mejor para LanceDB: +7,1% recall@10.
- Claim extraction fue la mÃ¡s dÃ©bil, pero mejorÃ³ Tantivy en 7,2%.
- Tantivy + summary alcanzÃ³ recall@10 0,799, cercano al baseline LanceDB 0,830.

## ConclusiÃ³n

El enrichment puede cerrar parte de la brecha lexical-vectorial. Debe permanecer como representaciÃ³n derivada, selectiva y separable del `canonical_text`.

## RecomendaciÃ³n

Aplicar enrichment sÃ³lo a chunks/documentos donde la mejora esperada justifique el coste y registrar siempre fingerprint, modelo e input hash.

## Provenance

Artefacto histÃ³rico (`outputs/experiments/E9-experiment/benchmark_report.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runners reproducibles en `scripts/benchmarks/run_enrichment_*.py`. Relacionado: [PM-002].
