---
id: RES-003
category: research
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [eks, agentic_runtime, knowledge_runtime, mcp, configuration]
tags: [engineering-memory, dev-time, runtime, separation, provenance]
related: [PAT-001, RES-002]
supersedes: null
superseded_by: null
---

# RES-003 — Separación EKS, Knowledge runtime y Agent Runtime

## Tema

Definir cómo conviven la memoria de ingeniería, el conocimiento del corpus y el runtime que lo consume.

## Modelo

```text
EKS dev-time
  decisiones, experimentos, benchmarks, patterns, research
             ↓
      Devin / Cascade

IPA Knowledge runtime
  documentos, chunks, índices, enrichment, provenance
             ↓
      Reporter / Agent Runtime / Tutor
```

## Conclusiones

- EKS no debe ser una segunda base vectorial del corpus.
- El Knowledge runtime no debe guardar decisiones del proceso de desarrollo.
- El Agent Runtime puede reutilizar contratos y adapters de IPA, pero no debe leer EKS para resolver consultas de usuario.
- El MCP EKS debe ser un servidor separado, local y read-only.
- ADRs, si aparecen, siguen en `docs/adr/`; la experiencia de ingeniería se registra en `knowledge/`.

## Estado

Aceptado como boundary de implementación EKS V1. Los detalles de memoria episódica, identidad y navegación permanecen abiertos y requieren experimentos separados.
