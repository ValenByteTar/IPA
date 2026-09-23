---
id: DEC-006
category: decision
status: accepted
created: 2026-09-09
updated: 2026-09-23
author: human
components: [agent_core, task_planner, strategic_memory, skill_library, uncertainty, user_model, system_tools, agent_identity]
tags: [cognitive-layer, planning, strategic-memory, skills, uncertainty, user-model, autonomy, bounded]
related: [DEC-002, DEC-005, RES-005, RES-006]
supersedes: null
superseded_by: null
evidence: ["src/ipa/agent/task_planner.py", "tests/test_cognitive_layer.py"]
affects: ["src/ipa/agent/task_planner.py", "src/ipa/agent/strategic_memory.py", "src/ipa/agent/skill_library.py", "src/ipa/agent/uncertainty.py", "src/ipa/agent/user_model.py", "configs/agent_identity.yaml"]
---

# DEC-006 — Capa cognitiva del agente

## Contexto

El agente personal (Fase 0-3) tenía base arquitectónica correcta — DocumentStore
canónico, tools determinísticas, promotion policy, registry unificado — pero
le faltaba la capa cognitiva: planificación, memoria estratégica, skills
dinámicas, calibración de incertidumbre, y user model transversal. El agente
era reactivo (usuario pide → agente responde con tools), no autónomo ni
adaptativo.

El bound de 3 tools/turno (DEC-005) es correcto para el LLM actual
(Qwen3.5-9B 3.0bpw) en chat interactivo, pero castra la autonomía de
horizonte largo. Subir el bound no funciona: el 9B degrada en cadenas largas
(RES-005). La solución es cambiar la unidad de bound: de "por turno" a "por
tarea con budget", donde la tarea es planificada y persistida fuera del
contexto del LLM.

## Decisión

Implementar 5 módulos cognitivos + 1 user model transversal, todos con el
mismo patrón: inferencia determinística (sin VRAM, idle Level 1) + inferencia
LLM opcional (idle Level 2) + gate humano (propuestas pending → approve →
active). Las capas active se inyectan en el system prompt dinámicamente.

### 1. Task planner + persistent task queue (puntos 2+3)

`src/ipa/agent/task_planner.py`:

- `TaskStore` (SQLite, persistente, resumible): tasks, subtasks, subtask_results.
- `Planner`: 1 generación LLM produce plan JSON → fallback a plantilla
  determinística si JSON inválido. El LLM nunca ejecuta tools — solo planifica.
- `TaskExecutor`: loop sobre sub-tasks, cada uno con el bound de 3 tools
  existente. Persiste progreso después de cada sub-task. Resumible.
- Budget: max 6 sub-tasks × 3 tools = 18 tools/tarea (vs 3 tools/turno).
- Tools: `plan_task`, `list_tasks`, `get_task`, `resume_task`.

El 9B no sostiene un plan de 20 pasos en su context window. Pero sí puede
generar un plan corto (una generación) y ejecutar cada sub-task por separado
(contexto corto). El plan vive en SQLite, no en el contexto del LLM.

### 2. Strategic memory (punto 4)

`src/ipa/agent/strategic_memory.py`:

- `StrategicMemoryStore` (SQLite, append-only proposals): principles.
- `StrategicReflector`: detecta tool_patterns (secuencias repetidas),
  query_noise (queries que dan 0 resultados), response_style (preferencia
  de longitud). Determinístico. + LLM opcional para principios abstractos.
- Gate: propuestas pending → humano approve → active → se inyectan en
  system prompt.

La memoria episódica graba qué pasó. La consolidación de sesiones resume
conversaciones. La strategic memory extrae PRINCIPIOS durables: "después de
research_topic, siempre search_corpus antes de compile_report".

### 3. Skill library dinámica (punto 5)

`src/ipa/agent/skill_library.py`:

- `SkillLibraryStore` (SQLite, append-only proposals): skills.
- `SkillDetector`: cuenta secuencias de tool_calls repetidas (>= 3
  ocurrencias, >= 2 tools por secuencia). Propone skills.
- Skills approved se inyectan en system prompt junto con las estáticas
  del YAML.

Las skills del YAML son estáticas. La skill library aprende nuevas del uso:
"hice search_corpus + compile_report 5 veces, hagamos eso una skill".

### 4. Uncertainty + active research agenda (punto 6)

`src/ipa/agent/uncertainty.py`:

- `UncertaintyStore` (SQLite): topic_confidence (EMA sobre scores),
  research_proposals.
- `UncertaintyTracker`: hook en search_corpus/compile_report/research_topic
  para trackear confianza por tópico.
- `ActiveResearchAgenda`: tópicos con confidence < 0.4 → propone research
  (pending → gate humano).
- Tool: `list_research_agenda`.

El agente no sabe qué no sabe. Esta capa trackea confianza y propone
investigación activa: "noté que sé poco sobre X, ¿investigo?".

### 5. User model transversal (punto 8)

`src/ipa/agent/user_model.py`:

- `UserModelStore` (SQLite): user_goals, user_interests, user_preferences,
  user_facts.
- `UserModelInferer`: intereses por frecuencia de queries, goals por tareas
  recurrentes, preferencias de longitud. Determinístico. + LLM opcional.
- Gate: inferencias pending → humano approve → active → se inyectan en
  system prompt.
- Tools: `get_user_profile`, `set_user_goal`, `set_user_interest`.

El user model hoy vive en tutor.db (acoplado al rol Tutor, solo mastery
pedagógica). El UserModelStore es transversal: todos los roles lo leen.
El rol Tutor sigue escribiendo mastery en tutor.db, pero lee user_model.db
para contexto.

### System prompt injection

`Identity.system_prompt()` ahora incluye 4 capas dinámicas (aditivas,
independientes, renderizan vacío si no hay datos):

1. User model (goals, interests, preferences, facts)
2. Strategic principles (aprendidos del uso)
3. Learned skills (flujos frecuentes)
4. Uncertainty topics (baja confianza)

Cada capa se wrappea en try/except: una falla en una capa nunca rompe el
system prompt. Fresh install → prompt sin capas dinámicas (no-op).

## Invariantes

- **Gate humano**: todas las inferencias (principles, skills, goals,
  research proposals) son pending → humano approve → active. Nunca
  auto-aplicadas. Patrón ConsolidationStore.
- **Determinístico por defecto**: la inferencia corre en idle Level 1 sin
  VRAM. La inferencia LLM es opcional (idle Level 2) y siempre gated.
- **Append-only**: proposals nunca se borran. Las rejected quedan como
  audit trail.
- **Resumibilidad**: TaskStore persiste current_subtask. "seguí lo de ayer"
  resume desde el último sub-task completado.
- **No-VRAM**: toda la capa cognitiva (excepto inferencia LLM opcional)
  corre en CPU. No compite con el chat por VRAM.
- **Bound preservado**: el bound de 3 tools/turno del chat interactivo no
  se cambia. La planificación cambia la unidad de bound a "por tarea con
  budget", no sube el bound por turno.

## Consecuencias

- El agente puede ahora: planificar tareas multi-paso, aprender del uso
  (principles + skills), trackear su propia incertidumbre, y adaptar
  respuestas al user model transversal.
- La autonomía de horizonte largo se habilita vía plan_task: el agente
  descompone el goal en sub-tasks y los ejecuta en background, sin volver
  al usuario cada 3 tools.
- El system prompt se enriquece dinámicamente: el agente "sabe" qué le
  interesa al usuario, qué principios aprendió, qué skills compose, y
  dónde tiene baja confianza.
- La complejidad nueva es modular: 5 módulos independientes, cada uno con
  su store SQLite, su inferer, y su gate. Fallo en uno no rompe los otros.

## No implementado (ver RES-006)

- Subagents paralelos: no viables en la RTX 4050 (6GB VRAM, no hay para 2+
  modelos 9B). Re-evaluar cuando cambie el hardware.
