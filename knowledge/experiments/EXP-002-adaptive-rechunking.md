---
id: EXP-002
category: experiment
status: accepted
created: 2026-09-05
updated: 2026-09-06
author: human
components: [chunking, retrieval, ingestion]
tags: [adaptive, rechunking, lexical-density, fallback]
related: [BM-004, PAT-002]
supersedes: null
superseded_by: null
---

# EXP-002 â€” Adaptive rechunking

## HipÃ³tesis

Reagrupar sÃ³lo documentos con baja densidad lexical puede reducir chunks problemÃ¡ticos sin degradar recall documental.

## ConfiguraciÃ³n

- Corpus analizado: 720 documentos.
- ActivaciÃ³n: mÃ¡s del 30% de chunks de un documento con lexical density menor que 0,4.
- Estrategia: merge adyacente y fallback recursive cuando fuese necesario.
- Artefacto: `outputs/experiments/E5-adaptive/retrieval/adaptive_retrieval_comparison.json`.

## Resultados

- 30 documentos activaron el proceso.
- Se realizaron 181 merges y se generaron 97 chunks de fallback.
- El delta total fue de -153 chunks.
- Recall documental@10 se mantuvo en 1,0 para el corpus activado.

## ConclusiÃ³n

La estrategia es segura para el corpus probado y puede usarse como operaciÃ³n derivada selectiva. No debe reemplazar automÃ¡ticamente el baseline fixed-window en todo el corpus.

## Provenance

Artefacto histÃ³rico (`outputs/experiments/E5-adaptive/retrieval/adaptive_retrieval_comparison.json`) no disponible localmente (ver PM-002). Conclusiones retenidas; runners reproducibles en `scripts/benchmarks/run_adaptive_rechunk.py` y `compare_adaptive_retrieval.py`. Relacionado: [PM-002].
