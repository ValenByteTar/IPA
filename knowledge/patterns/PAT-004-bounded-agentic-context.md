---
id: PAT-004
category: pattern
status: proposed
created: 2026-09-05
updated: 2026-09-05
author: human
components: [agentic_runtime, planner, retrieval, context, evaluation]
tags: [bounded-loop, query-ir, evidence-set, context-package, citations]
related: [EXP-003]
supersedes: null
superseded_by: null
---

# PAT-004 — Context agentivo acotado

## Problema

Un Deep Dive puede mezclar planning, retrieval, context building y generación, ocultando qué información entró realmente al modelo.

## Solución

Separar explícitamente:

```text
pregunta
  → QueryIR
  → EvidenceSet
  → ContextPackage
  → generación
```

El planner no recupera; retrieval no construye prompts; context builder no conoce el modelo; generación no consulta directamente el store. Los budgets y la trazabilidad acompañan cada ejecución.

## Trade-offs

Introduce contratos y adapters, pero permite replay, tests aislados, límites de contexto y validación de citas.

## Estado

Propuesto hasta completar evaluación end-to-end del runtime `agentic_v1`.
