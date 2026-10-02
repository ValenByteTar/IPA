---
id: DEC-012
category: decision
status: draft
created: 2026-09-25
updated: 2026-09-26
author: Valen + Devin
components: [tutor, agent_core, dashboard]
tags: [roadmap, workbench, tutor, depth, refinement]
related: [DEC-002, DEC-006, DEC-009, BM-006, RES-005, PAT-004]
supersedes: null
superseded_by: null
affects: [src/ipa/tutor/**, src/ipa/dashboard/**, contracts/**, web/static/**, src/ipa/agent/query_gate.py, src/ipa/agent/memory_store.py, src/ipa/agent/session_consolidator.py]
evidence: ["src/ipa/tutor/tutor_workbench.py", "src/ipa/tutor/tutor_expansion.py", "contracts/roadmap.schema.json", "contracts/roadmap_expansion.schema.json", "tests/test_tutor_workbench.py"]
author_model: GLM-5.3 Flash Max (Devin)
trigger: permit:PW-20260925-07
---

# DEC-012 — Estacion de montaje de Roadmaps: draft editable, refinamiento 3-capas y dimension de profundidad

## Contexto

La pestaña Roadmaps era un read-model frágil: el roadmap se creaba en una
única generación LLM (propose → gate) y cualquier ajuste era regenerar TODO
el roadmap con feedback en lenguaje natural, perdiendo unidades buenas. El
contrato v1 no tenía estado mutable pre-propuesta (units 3-7 obligatorias,
sin draft) y las lecciones no inyectaban el material del corpus (el 9B
enseñaba de memoria — verificado en sesión 2026-09-25: las respuestas sobre
"Jev" eran elaboración sin evidencia, con sources de quality_score 0.0).

El usuario pidió: (1) una estación de montaje donde ver/tocar/romper/crear/
debatir roadmaps, (2) acceso implícito desde el chat general sin mencionar
el roadmap, (3) una dimensión de profundidad para cobertura de
enriquecimiento, (4) refinamiento en múltiples pasadas.

## Decisión

1. **Contrato v2** (`roadmap.schema.json`): nuevo estado `draft` (0-50
   unidades, ungrounded permitido) que se congela en spine 3-7 grounded
   (`frozenUnit`); `stages[]` opcional (eje Z); `validation` adjunta (P2);
   `field_origins` por unidad. Nuevo contrato `roadmap_expansion.schema.json`
   (dimensión Y: niveles L0 overview / L1 nuclear / L2 claims / L3 vecindario,
   patrón PAT-004 EvidenceSet).

2. **Estación de montaje** (`tutor_workbench.py`): CRUD determinístico de
   drafts (add/remove/reorder/edit_unit/set_stages, cada op con
   field_origins user), `freeze_draft` (bloqueado con findings de severidad
   error), `reopen_draft` (solo proposed → draft). El gate humano
   (approve_roadmap) sigue siendo el único camino a active.

3. **Refinamiento 3-capas** (P1→P2→P3): LLM genera → validación
   DETERMINÍSTICA (grounding, cobertura, esfuerzo, conteo) → LLM revisor con
   contexto corto (solo hallazgos + unidades afectadas, max 3). NO es
   deliberación multi-agente (DEC-009 rechazada con daño neto medido): la
   capa crítica es código, y P3 entra con contexto acotado (RES-005).

4. **Navegación determinística**: posición `(unit_id, depth_level)` en
   SQLite (`roadmap_position`) — "¿dónde quedé?" es un SELECT, nunca
   inferencia del LLM. Expansión de profundidad determinística
   (`tutor_expansion.py`) sobre DocumentStore + topic_clusters: el LLM solo
   renderea el material.

   Enmienda 2026-09-26: el keying de `unit_progress`, `roadmap_position` y
   `unit_summaries` pasó de `unit_order` a `unit_id`. Razón: el orden es
   presentación (cambia al reordenar en la estación), la identidad es el
   `unit_id` — que ahora deriva de `hash(goal_id, concept_id)` SIN el
   índice posicional, así una revisión que reordena conserva la identidad.
   Consecuencia directa: `TutorStore.inherit_progress()` copia progreso,
   posición, summaries y expansiones L0-L3 a la versión nueva al activarla
   (solo unidades que sobrevivieron — `activate_roadmap` la invoca cuando
   hay `previous_roadmap_id`). Migración lectora `_migrate_unit_keying()`:
   remapea filas viejas via el payload del roadmap; las que no resuelven se
   descartan (eran huérfanas de hecho). DB real migrada: 5 filas de
   progreso + 1 summary preservadas. Además: `remove_unit` borra las
   expansiones del unit (no quedan huérfanas) y `edit_unit` con
   `concept_id` nuevo las marca `stale` (la evidencia describe el material
   viejo hasta re-expandir).

5. **Acceso implícito desde chat general**: `classify_message` gana el kind
   `roadmap` (regex específica, antes que memory); el estado se resuelve
   server-side (patrón recall_memory — el 9B no elige tools). Con roadmap
   enfocado, una línea compacta determinística entra SIEMPRE al system
   prompt del chat general.

6. **UI**: pestaña Roadmaps → estación de montaje (editor de draft con
   quitar/reordenar/editar/agregar, hallazgos P2 visibles, botones
   Refinar/Validar/Congelar) + grid de profundidad unidades × L0-L3 con
   expansión on-demand.

Fuera de alcance: deliberación multi-agente (DEC-009), stages como
roadmaps encadenados (se eligió sub-estructura), benchmark 3-pasadas-LLM
(opcional post-launch si se quiere reabrir DEC-009).

## Consecuencias

- Mejora: montaje granular sin regeneración total; lecciones anclables a
  evidencia real; escala a roadmaps grandes (50 unidades en draft, stages);
  el 9B queda confinado a proponer/renderizar (BM-006: clasifica bien,
  narrativa libre degrada).
- Coste: schema v2 (migración lectora verificada: 8/8 roadmaps existentes
  válidos); L2 (claims) depende del backlog de enriquecimiento (~330K chunks
  pendientes — sin claims el nivel falla con error accionable, honesto);
  app.js monolítico creció (~150 líneas).
- Reversión: los campos nuevos son opcionales; los roadmaps v1 siguen
  válidos. Revertir = restaurar schema v1 + eliminar tutor_workbench/
  tutor_expansion + endpoints.

## Evidencia

- `tests/test_tutor_workbench.py`: 19 tests (CRUD, invariantes, P2, freeze,
  P3 con contexto corto verificado en el prompt del FakeProvider, cleanup
  de expansiones huérfanas/stale, `inherit_progress` entre versiones, y
  migración `unit_order`→`unit_id` con backfill desde payload).
- Suite completa: 1208 passed, 1 skipped (2026-09-26, era 1204 pre-keying).
- End-to-end en vivo: draft creado vía API, 3 unidades grounded con docs
  reales, validate sin findings, freeze → proposed, expansión L0 generada
  desde `doc:63c60522b84265fb` (agentes.ai, 3800 chars).
- Bug colateral corregido: `scripts/hooks/permit_guard.py` moría con
  UnicodeEncodeError (cp1252) al inyectar records EKS con →/— en títulos —
  la inyección nunca llegaba a la UI. Fix: stdout UTF-8 forzado.

## Alcance

Toca la frontera tutor (estado pedagógico) y el gate del chat general
(query_gate). No toca providers ni VRAM (EXP-008 intacto). Se revierte o
supersede con un DEC nuevo que referencie esta evidencia.
