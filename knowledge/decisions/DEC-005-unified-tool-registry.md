---
id: DEC-005
category: decision
status: accepted
created: 2026-09-12
updated: 2026-09-23
author: agent
components: [agent_core, dashboard]
tags: [tools, system-tools, registry, tool-catalog, bounded-loop, chat]
related: [DEC-002, DEC-004]
supersedes: null
superseded_by: null
evidence: ["src/ipa/agent/system_tools.py", "tests/test_system_tools.py"]
affects: ["src/ipa/agent/system_tools.py", "src/ipa/agent/agent_tools.py", "src/ipa/mcp/**"]
---

# DEC-005 — Registry unificado de tools del agente

## Contexto

El agente tenía dos mundos de tools desconectados:

- `agent_tools.py`: tools determinísticas del corpus (`search_corpus`, `list_topics`,
  `get_topic_info`, `recall_conversation`) más los executors (`research_topic`,
  `compile_report`) que estaban en `TOOL_NAMES` pero no en `_TOOL_IMPLEMENTATIONS` —
  `execute_tool()` los rechazaba como "unknown tool".
- `system_tools.py`: tools visibles para el LLM del chat via `TOOL_CATALOG` hardcodeado.
  El catálogo estaba desincronizado: listaba `run_report` (roto), no listaba
  `compile_report` (recién creada), y ninguna tool de retrieval era visible para el chat.

Consecuencias: el chat no podía buscar en el corpus (el prompt pide citar `[n]` pero no
había forma de obtener hits), `run_pipeline`/`run_ingestion` eran duplicados tras la
remoción del Reporter, `promote_to_main` operaba sobre el Reporter batch (obsoleto con
la promoción por política de DEC-003), y el fuzzy match de `parse_tool_marker` tenía
una colisión no determinística entre `get_report` y `compile_report` (mismo key_part
`"report"`, iteración sobre frozenset con orden arbitrario). Además había dos clases
`SystemToolResult` definidas — la segunda pisaba a la primera.

## Decisión

1. **Registry unificado** (`SystemToolSpec` en `system_tools.py`): cada capability es
   un spec `(name, description, args_doc, fn, chat_visible)`. `SYSTEM_TOOL_NAMES` y
   `_SYSTEM_IMPLEMENTATIONS` se derivan del registry.

2. **`TOOL_CATALOG` generado** desde los specs con `chat_visible=True` — el prompt del
   LLM no puede desincronizarse del registry nunca más.

3. **Tools nuevas visibles para el chat**:
   - `search_corpus` — retrieval híbrido con hits numerados `[n]` (cierra el gap de
     evidencia/citas en el chat general).
   - `list_topics` — tópicos emergentes del TopicClusterStore.
   - `list_promotions` — cola de promoción (read-only; la promoción corre sola en idle).
   - `research_topic` — investigación web async (thread + `research_progress.json` +
     `RESEARCH_WATCH` que entrega episodio-resumen a la sesión al terminar, mismo
     patrón que `PIPELINE_WATCH`).

4. **Colapsos**: `run_ingestion` es la única tool de ingesta; `run_pipeline` queda como
   alias oculto dispatchable (no aparece en el catálogo). `promote_to_main` removida —
   la promoción es política continua (DEC-003), no una acción del modelo.

5. **Dispatch de executors en `agent_tools.py`**: `execute_tool()` rutea
   `compile_report` y `research_topic` a sus executors (que emiten sus propios
   contratos) y envuelve errores de validación en `ToolCall`/`ToolResult` con
   `status="failed"` — el registry es completo y consistente.

6. **Fuzzy parsing determinístico**: pases (a) contención de nombre completo,
   (b) key_part única, (c) scoring por nombre completo — siempre iterando
   `sorted(SYSTEM_TOOL_NAMES)`. `[REPORT]` → `get_report`, `[TOPIC]` → `research_topic`,
   `[TOPICS]` → `list_topics`. `[PROMOTE_TO_MAIN]` garbled resuelve a la tool real más
   cercana en vez de fallar.

7. **Bounded tool loop** (max 3 rondas por turno) en el handler de streaming:
   reemplaza el protocolo de una sola ronda. El modelo puede encadenar
   `search_corpus` → `compile_report` en un turno. Guard anti-duplicados: la misma
   `(name, args)` no se ejecuta dos veces. Las tools async arman watchers.

8. **Skills prompt-level** en `agent_identity.yaml`: sección `skills:` documenta
   flujos típicos (evidencia, reporte, investigación web, ingesta) renderizados en
   el system prompt. `capabilities:` actualizada al set real.

## Invariantes preservados

- El LLM solo SELECCIONA tools; la ejecución es determinística (PAT-004 budgets en research).
- `compile_report`/`research_topic` emiten los mismos contratos ToolCall/ToolResult
  por cualquier vía de invocación (registry, executor directo, system tool).
- Las tools async no bloquean el chat; los watchers respetan `CHAT_BUSY`.
- `research_topic` usa `HeuristicJudge` (sin VRAM) — seguro junto al chat con modelo cargado.
- La promoción sigue siendo política + cola + executor; el agente solo la lee.

## Tests

`tests/test_system_tools.py` (24): consistencia del registry, catálogo generado,
fuzzy determinístico (10 iteraciones), dispatch de executors, paths de validación.
