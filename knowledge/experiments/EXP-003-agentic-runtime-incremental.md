---
id: EXP-003
category: experiment
status: proposed
created: 2026-09-05
updated: 2026-09-05
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

## Métricas

- doc hit@K;
- relevancia de chunks;
- validez de citas;
- tasa de decline correcto;
- latencia p50/p95;
- igualdad de respuesta cuando el flag está desactivado;
- utilidad humana del Deep Dive.

## Estado

Propuesto. Los contratos iniciales ya existen en `src/res023_lab/agentic_contracts.py`, pero todavía se requiere evaluación end-to-end antes de promover una arquitectura agentiva más amplia.

## Recomendación

Ejecutar primero una comparación aislada. No agregar Policy, Controller, memoria persistente ni loops ReAct hasta demostrar valor incremental y ownership claro.
