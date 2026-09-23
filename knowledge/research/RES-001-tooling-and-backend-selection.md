---
id: RES-001
category: research
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [retrieval, lexical_index, vector_index, parsing, chunking, configuration]
tags: [tool-selection, benchmark, adapter, local-first, workload]
related: [BM-001, BM-002, BM-003, BM-004, BM-005]
supersedes: null
superseded_by: null
affects: ["docs/policies/tool-selection.md", "src/ipa/agent/system_tools.py"]
---

# RES-001 — Selección de herramientas por workload

## Tema

Cómo seleccionar implementaciones para IPA sin convertir una victoria puntual en un default universal.

## Fuentes

- `docs/TOOL_DECISION_FRAMEWORK.md`
- `docs/DECISION_LOG.md`
- benchmarks E3, E5, E6, E7 y E10 en `outputs/experiments/`.

## Comparativa

- Tantivy favorece indexación y búsqueda lexical de escala.
- LanceDB favorece el workload vectorial medido.
- PyMuPDF favorece el fast path PDF born-digital.
- Fixed-window favorece throughput y estabilidad.
- Hybrid retrieval es apropiado cuando la consulta exige semántica y no sólo términos exactos.

## Takeaways

La unidad de selección debe ser `capability + workload`, no “la mejor herramienta” en abstracto. Cada implementación debe estar detrás de un adapter y promocionarse mediante contract compliance, provenance, idempotencia, recovery, coste y métricas de calidad.
