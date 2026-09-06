---
id: RES-002
category: research
status: proposed
created: 2026-09-05
updated: 2026-09-05
author: human
components: [agentic_runtime, reporter, tutor, retrieval, context, mcp]
tags: [agentic-rag, crosswalk, ownership, boundaries, adapters]
related: [EXP-003, PAT-004]
supersedes: null
superseded_by: null
---

# RES-002 — Crosswalk de boundaries agentivos

## Tema

Comparar patrones de AgenticRAG con la implementación actual de IPA sin copiar código ni convertir decisiones externas en autoridad local.

## Observaciones

- IPA es principalmente adquisición, materialización, índices, enrichment, jobs y exposición del corpus.
- Reporter es una capability analítica aislada con tópicos, curación, deep dive y promoción.
- Los contratos `QueryIR`, `EvidenceSet` y `ContextPackage` son una primera frontera local para investigación agentiva.
- Tutor posee estado pedagógico propio y no debe mezclarse con EKS ni con ingestion.
- Qwen/ExLlama es una implementación de provider; no debe definir ownership del runtime.

## Takeaways

Los patrones externos deben incorporarse sólo mediante contracts, adapters, tests y evidencia local. El planner, retrieval, context, generation, evaluation y orchestration deben conservar responsabilidades independientes. Este research no autoriza todavía Policy, Controller, memoria persistente ni una arquitectura Personal AGI.

## Gaps

- falta benchmark end-to-end del runtime agentivo;
- falta definir ownership definitivo de memoria conversacional;
- falta demostrar valor incremental de navegación horizontal y multi-hop;
- falta decidir si alguna frontera futura merece ADR local.
