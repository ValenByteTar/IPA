---
id: PAT-004
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [agentic_runtime, planner, retrieval, context, evaluation]
tags: [bounded-loop, query-ir, evidence-set, context-package, citations]
related: [EXP-003]
supersedes: null
superseded_by: null
evidence: ["src/ipa/agent/research_executor.py", "tests/test_agentic_runtime.py"]
affects: ["src/ipa/agentic/**", "src/ipa/agent/research_executor.py"]
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

Aceptado (2026-09-06) tras la evaluación end-to-end de EXP-003: calidad de retrieval idéntica al camino legacy, citation map con hash verificado al 100%, overhead de latencia ~7-9% y flag apagado sin filtración del runtime agentivo. Ver `knowledge/experiments/EXP-003-agentic-runtime-incremental.md`.
