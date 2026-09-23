---
id: DEC-002
category: decision
status: accepted
created: 2026-09-06
updated: 2026-09-23
author: human
components: [agentic_runtime, agent_core, memory, configuration, eks]
tags: [agent-core, user-model, ownership, sessions, episodes, identity, boundaries]
related: [RES-002, RES-003, PAT-004, EXP-003]
supersedes: null
superseded_by: null
affects: ["src/ipa/agent/**", "configs/agent_identity.yaml", "contracts/**"]
---

# DEC-002 — Ownership del user model unificado y boundaries del agent core

## Contexto

La hoja de ruta `docs/plans/agent-core-roadmap.md` fusiona el pseudo-tutor con la horizontalidad sobre un núcleo agentivo de tres capas (núcleo / superficies / roles). RES-002 dejó abiertos el ownership de la memoria conversacional y los boundaries del runtime agentivo. Esta decisión los cierra para la Fase 0, con las decisiones aprobadas explícitamente por el usuario.

## Decisiones (aprobadas por el usuario, 2026-09-06)

1. **Store del agente**: `outputs/agent/` con subestructura auditable —
   `agent.db` (SQLite plano, canónico, append-only), `vector/` (índice derivado
   de episodios, Fase 3), `exports/` (dumps de auditoría). El store está
   **fuera** de E12-corpus: el agente es omnipresente, no corpus-bound; acoplar
   su memoria a un corpus repetiría el error de anclarlo al dashboard.
2. **Modelo sesión/episodio**: episodios planos con `session_id`, append-only y
   queryables (patrón `trace_log` / SQL de Horizontalidad). La búsqueda
   vectorial sobre episodios es **representación derivada** (patrón
   DocumentStore → LanceDB) y se agrega en Fase 3 cuando exista su consumidor
   (`topic_cluster_id` + recall semántico); FTS sobre episodios es el punto
   medio opcional de Fase 1. Nada de índices derivados sin consumidor.
3. **Identidad**: `configs/agent_identity.yaml` — es configuración, no dato;
   queda en `configs/` junto a las demás y su auditoría es el historial de git.
   Cada episodio registra `identity_hash` (sha256 del YAML activo) para vincular
   memoria con la versión de identidad vigente en el turno.
4. **Roles v1**: campo `role` en sesiones/episodios + policies en código. El
   scoping formal de estado por rol llega en Fase 2 con el Tutor. No
   sobre-ingenierizar en Fase 0.
5. **IDs y vocabulary**: prefijos `agent_session:`/`agent_episode:` bajo el
   patrón identifier de `tutor_common`; registros `AgentSession`/`AgentEpisode`
   en `contract_vocabulary.json` con sus invariants.

## Boundary del agent core

El núcleo (`ipa/agent/`) posee identidad, sesiones, memoria episódica y (en
fases posteriores) tools y policies de rol. **No posee** corpus, índices ni
ingestión — consume IPA vía adapters (PAT-004: QueryIR → EvidenceSet →
ContextPackage). Las superficies (CLI, dashboard) son clientes finos que abren
sesiones; ninguna define personalidad ni guarda estado propio del agente.

## Extracts (desviación consciente del roadmap)

Los extractos consolidados (`agent_memory_extracts`) **no** se crean en Fase 0:
no se crea tabla sin contrato que la autorice (contract-first). Su contrato
propio llega en Fase 2/3 junto con la consolidación semi-automática con
aprobación humana.

## Consecuencias

- La memoria del agente sobrevive a cambios de modelo, superficies y corpus
  (estado propio, ubicación propia).
- La privacidad queda crispada por el layout: episodios (conversaciones
  personales) viven separados del corpus y nunca salen del equipo.
- Cambiar de modelo/superficie no reescribe memoria ni contratos; el provider
  ya es reemplazable (DEC-001) y ahora el estado también es propio.
- Los roles nuevos (Fase 2+) extienden el enum `role` por supersede del schema.

## Rollback

El store es un archivo SQLite propio: backup/copia = auditoría completa. Los
contratos evolucionan por `supersedes`; el store puede reconstruirse desde
`exports/` si se corrompiera (y viceversa).

## Condiciones de aceptación

- [x] decisiones 1-5 aprobadas por el usuario (2026-09-06);
- [x] schemas `agent_common`/`agent_session`/`agent_episode` congelados y
  registrados en el vocabulary;
- [x] gate de omnipresencia verificado (2026-09-06): sesión escrita desde un
  proceso y leída desde otro — `tests/test_agent_core.py` (10 tests, incluido
  el gate entre procesos reales vía subprocess).
