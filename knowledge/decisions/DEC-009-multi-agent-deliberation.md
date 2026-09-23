---
id: DEC-009
category: decision
status: rejected
created: 2026-09-23
updated: 2026-09-23
author: human
components: [agent_core, providers, evaluation, performance]
tags: [multi-agent, deliberation, subagents, vram, rejected, evidence]
related: [EXP-004, BM-006, RES-006, RES-005]
supersedes: null
superseded_by: null
affects: ["src/ipa/agent/**", "src/ipa/providers/**"]
---

# DEC-009 — Deliberación multi-agente como mecanismo de calidad

## Contexto

Se evaluó usar deliberación multi-agente (múltiples pasadas/roles del modelo
que se critican entre sí) para elevar la calidad del Tutor y del chat sobre
el Qwen3.5-9B local.

## Propuesta rechazada

Correr deliberación multi-agente o subagents LLM paralelos como mecanismo de
calidad del sistema.

## Evidencia del rechazo

- **Daño neto medido**: la deliberación rindió -3/+3 con daño real en el
  benchmark de engines (EXP-004, sección de deliberación; resumen en
  `docs/plans/agent-core-roadmap.md` → "Lo que NO entra").
- **Imposibilidad física**: RES-006 documenta que en 6 GB de VRAM no caben
  dos modelos 9B; el time-slicing no es paralelismo real y degrada el
  throughput total.
- El modo `independent` (una generación, sin deliberación) es el adoptado:
  assessment con abstención 0.81-0.89 medida (BM-006).

## Consecuencias

- No reintentar deliberación multi-agente ni subagents LLM **sin evidencia
  nueva**: cambio de hardware (VRAM para 2+ modelos), cambio de modelo
  estrella, o un benchmark que muestre ganancia neta.
- El registro existe para que la propuesta no se re-litigue desde cero:
  la evidencia del rechazo está enlazada, no es una intuición.
- RES-006 permanece como el research de fondo (qué sí es viable sin VRAM
  extra: subagents determinísticos, roles secuenciales, research async).

## Alcance

Cierra la dirección multi-agente para el runtime actual. No prohíbe
orquestación determinística ni paralelismo de I/O (scrape/recolección), que
sí están en uso.
