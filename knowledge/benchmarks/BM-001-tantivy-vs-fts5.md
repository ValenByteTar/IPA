---
id: BM-001
category: benchmark
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [lexical_index, retrieval, performance]
tags: [tantivy, fts5, bm25, indexing, latency, disk]
related: [RES-001]
supersedes: null
superseded_by: null
---

# BM-001 — Tantivy vs FTS5

## Objetivo

Comparar los backends lexicales sobre el mismo workload y congelar el resultado para evitar elegir un backend por intuición.

## Entorno

- Corpus: 166k chunks.
- Backends: SQLite FTS5 y Tantivy.
- Métricas: throughput de indexación, latencia p50 de consulta y tamaño en disco.
- Artefacto: `outputs/experiments/E6-full/benchmark_report.json`.

## Resultados

| Métrica | FTS5 | Tantivy |
|---|---:|---:|
| Indexación | 182 chunks/s | 30.724 chunks/s |
| Latencia p50 | 1 ms | 0 ms |
| Disco | 1,1 GB | 139 MB |

## Conclusión

Tantivy fue preferido para el workload lexical de escala por throughput y tamaño de disco, manteniendo latencia comparable. FTS5 permanece como baseline y backend de referencia de Stage 1.

## Alcance

Este benchmark no demuestra que Tantivy sea superior para toda consulta semántica ni reemplaza la evaluación híbrida.
