---
id: EXP-003
category: experiment
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [agentic_runtime, planner, retrieval, context, reporter]
tags: [query-ir, evidence, context-package, deep-dive, agentic-v1]
related: [RES-002, PAT-004]
supersedes: null
superseded_by: null
---

# EXP-003 — Runtime agentivo incremental para Reporter

## Hipótesis

Introducir QueryIR, EvidenceHit, EvidenceSet y ContextPackage como adapters independientes puede mejorar la trazabilidad del Deep Dive sin cambiar el camino legacy cuando el feature flag está apagado.

## Configuración propuesta

- Entrada: pregunta o tópico del Reporter.
- Planner: determinista, sin llamada LLM.
- Retrieval: Tantivy y opcionalmente búsqueda híbrida.
- Context: presupuesto explícito y citation map.
- Comparación: `agentic_v1` contra el camino existente.

## Ejecución (2026-09-06)

Harness reproducible: `src/ipa/agentic/agentic_eval.py` + entrypoint `scripts/benchmarks/run_agentic_eval.py`. Corpus: `outputs/reporter/quality-check/optimized-llm-2026-08/corpus` (Reporter aislado con tantivy + document_store + reporter.db). Reporte: `outputs/experiments/E13-agentic/report.json`.

Diseño: 12 queries de ground truth determinista (términos distintivos por documento vía tf-idf, dos familias: term y title) + 3 queries sin sentido para decline. Ambos caminos con `provider=None` — la superficie medida es exactamente planner, retrieval, context, validación de citas y decline. Warm-up previo para no confundir la latencia con el cold-start del índice.

## Resultados

| Métrica | legacy | agentic_v1 |
|---|---:|---:|
| doc hit@5 | 1.0 | 1.0 |
| precision@5 | 0.833 | 0.833 |
| existencia de chunks en store | 1.0 | 1.0 |
| supported claim rate | 0.308 | 0.308 |
| citation hash match (citation map) | — (sin map) | **1.0** |
| decline correcto (3/3) | 1.0 | 1.0 |
| latencia p50 | 16.65 ms | 17.88 ms |
| latencia p95 | 17.58 ms | 19.17 ms |
| igualdad con flag apagado | — | **true (0 fallos)** |

## Lectura

- **Calidad de retrieval idéntica**: el planner determinista + ReporterRetriever no degrada hit@5 ni precision@5 frente al camino legacy en este corpus.
- **Trazabilidad nueva sin costo de calidad**: el agentic añade el citation map cerrado con hash de texto verificado al 100% contra el store; el legacy no tiene map por diseño.
- **Overhead despreciable**: +1.2 ms p50 / +1.6 ms p95 (~7-9%) por planificación explícita y construcción de contratos.
- **Flag apagado preservado**: dos corridas legacy son idénticas (evidence, answer, claims) y `runtime` permanece `None` — cero filtración del runtime agentivo.
- **Decline correcto**: las 3 queries sin sentido devuelven `sufficient_evidence=False` en ambos caminos.

## Limitaciones

- N pequeño: 12 queries GT sobre los documentos del corpus Reporter de quality-check; no representa la escala de E10 (166k chunks).
- `provider=None`: la calidad de la respuesta final y la utilidad humana del Deep Dive quedan fuera de esta evaluación (requieren LLM); lo medido es la superficie determinista que alimenta al modelo.
- La variante híbrida (`deep_dive_prepare` con LanceDB 3-way RRF) comparte los contratos pero no fue objeto de métricas separadas aquí.

## Estado

Aceptado: la hipótesis se confirma para la superficie determinista. Los contratos iniciales viven en `src/ipa/agentic/` y el flag `agentic` de `deep_dive` preserva el camino legacy. Promoción de una arquitectura agentiva más amplia (Policy, Controller, memoria persistente, loops ReAct) sigue sujeta a las condiciones de RES-002.

## Recomendación

- [x] Mantener como Experiment
- [ ] Congelar como Benchmark
- [ ] Crear Decision
- [ ] Proponer ADR
- [ ] Nothing
