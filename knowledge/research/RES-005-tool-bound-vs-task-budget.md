---
id: RES-005
category: research
status: proposed
created: 2026-09-09
updated: 2026-09-09
author: human
components: [agent_core, dashboard_api, system_tools, research_executor]
tags: [agentic-loop, autonomy, safety, budget, llm-constraints]
related: [DEC-005, RES-002, RES-006]
supersedes: null
superseded_by: null
---

# RES-005 — Bound por turno vs budget por tarea

## Tema

El loop de tools del chat del dashboard está acotado a `MAX_TOOL_ROUNDS = 3` por turno de usuario (`src/ipa/dashboard/api.py:894`). Este bound es seguro pero castra la autonomía del agente para investigación de horizonte largo. Este research documenta el problema, la restricción de hardware real, y por qué la solución obvia (subir el bound) no es viable con el LLM actual.

## Observaciones

### El bound actual y su razón de ser

El loop de streaming en `api.py` (líneas ~891-970) implementa:

- Máximo 3 tools por turno de usuario.
- Anti-duplicación: la misma `(tool_name, args)` se rechaza con contexto al modelo.
- Al alcanzar el límite, el modelo recibe `"límite de 3 herramientas por turno alcanzado"` y debe responder sin más tools.

El bound existe por dos razones válidas:

1. **Safety**: sin bound, un modelo que entre en loop (tool → resultado → misma tool → ...) consume VRAM y tiempo indefinidamente.
2. **Latencia de chat**: cada tool round es una generación extra. 3 rounds ya son ~30s con el 9B cuantizado. 20 rounds serían minutos por turno.

### El problema real

Una investigación seria necesita 20+ pasos:

```
buscar → leer resultados → refinar query → buscar más
  → descartar irrelevantes → profundizar en 2-3 fuentes
  → cruzar fuentes contradictorias → sintetizar → compilar reporte
```

Con 3 tools/turno, el agente **no puede** hacer esto en un solo turno. Vuelve al usuario después de 3 tools y el usuario tiene que decir "seguí" para que el agente haga 3 tools más. Eso no es autonomía — es tele-operación.

### La restricción de hardware

El modelo estrella es **Qwen3.5-9B EXL3 3.0bpw** corriendo en una RTX 4050 (sm_89, 6GB VRAM efectiva para el modelo). Características relevantes:

- **Context window limitada**: ~8K-16K tokens prácticos antes de degradación de calidad.
- **Calidad de razonamiento**: un 9B cuantizado a 3.0bpw razona bien para tareas cortas pero **degrada** en cadenas largas — acumula errores de planificación, olvida sub-goals, repite pasos.
- **VRAM contended**: el mismo modelo sirve chat + idle enrichment Level 2 + research judges. No hay VRAM para un segundo modelo que planifique en paralelo.
- **Latencia**: cada generación de ~200 tokens toma 3-8s. 20 rounds = 1-3 minutos solo en generación, sin contar tool execution.

### Por qué subir el bound a 20 no funciona

Aunque cambiáramos `MAX_TOOL_ROUNDS = 20`:

1. **El 9B pierde el hilo**: después de ~5-6 rounds, el contexto acumulado (system prompt + historial + 6 tool results) satura la window. El modelo empieza a ignorar instrucciones tempranas, repite tools, o alucina resultados.
2. **No hay planificación**: sin un planner que mantenga el árbol de sub-goals **fuera** del contexto del LLM, el modelo tiene que "recordar" qué falta hacer dentro de su context window. Un 9B no puede sostener eso.
3. **Costo de error**: si el modelo se equivoca en el paso 8 de 20, los pasos 9-20 trabajan sobre el error. Sin checkpoints de progreso, no hay forma de detectar y recuperar.
4. **Sin resumibilidad**: si el dashboard se cierra en el paso 12, se pierden los 12 pasos. No hay estado durable.

## Comparativa

| Enfoque | Bound | Autonomía | Viable con 9B? | Riesgo |
|---|---|---|---|---|
| **Actual: 3 tools/turno** | por turno | baja (tele-op) | sí | bajo |
| **Subir a 20 tools/turno** | por turno | media | **no** — degrada | alto (loop, alucinación) |
| **Budget por tarea + planner externo** | por tarea | alta | sí (con planner determinístico) | medio (complejidad nueva) |
| **Background research worker** | por tarea, async | alta | sí (sin bloquear chat) | medio |
| **Multi-agente (subagents)** | por subagent | muy alta | **no** — sin VRAM para 2+ modelos | alto (ver RES-006) |

## Takeaways

1. **El bound de 3 tools/turno es correcto para el LLM actual y el chat interactivo.** No se debe subir sin una capa de planificación externa.

2. **La solución no es subir el bound — es cambiar la unidad de bound.** De "por turno" a "por tarea con budget", donde la tarea es planificada y persistida **fuera** del contexto del LLM. Esto se trata en RES-002 (planner) y la discusión de planificación con el LLM actual.

3. **El 9B cuantizado es el cuello de botella real, no el bound.** Un modelo más grande (70B, o un 9B sin cuantizar con más context) podría sostener 10-15 rounds con calidad. Pero no es lo que tenemos. La arquitectura debe ser correcta **para el hardware real**, no para un hardware hipotético.

4. **El patrón correcto para el 9B es**: planner determinístico (no LLM) que descompone el goal en sub-tasks → el LLM ejecuta cada sub-task con su bound de 3 tools → el planner persiste progreso entre sub-tasks → el LLM no necesita "recordar" el plan completo, solo el sub-task actual.

5. **No implementar subagents paralelos** hasta tener VRAM para 2+ modelos. Ver RES-006.

## Gaps

- Falta implementar el planner + task queue (ver discusión de planificación).
- Falta definir el contrato de `Task` con `budget`, `state`, `subtasks`, `resume_point`.
- Falta decidir si el planner es determinístico (reglas) o usa el LLM en modo "plan-only" (una sola generación, sin tools).
- Falta benchmark de degradación del 9B a partir de qué round el modelo pierde el hilo (medir quality vs round count).
