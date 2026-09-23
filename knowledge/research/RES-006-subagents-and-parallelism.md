---
id: RES-006
category: research
status: accepted
created: 2026-09-09
updated: 2026-09-23
author: human
components: [agent_core, providers, research_executor]
tags: [subagents, parallelism, vram, multi-agent, architecture]
related: [RES-005, RES-002, DEC-002, DEC-009]
supersedes: null
superseded_by: null
evidence: ["docs/architecture/agent-runtime.md"]
affects: ["src/ipa/agent/research_executor.py", "src/ipa/providers/**"]
---

# RES-006 — Subagents y paralelismo agentivo

## Tema

Investigar el patrón de subagents (múltiples agentes explorando ángulos distintos en paralelo) como capa de orquestación sobre el agent core, evaluar su valor para IPA, y documentar por qué **no se implementa ahora** por restricción de VRAM.

## Fuentes

- Patrones de AgenticRAG multi-agent (ReAct, Reflector, Plan-and-Execute, LangGraph multi-agent).
- `src/ipa/providers/exl3_provider.py` — provider actual, single-model, continuous batching.
- `src/ipa/agent/research_executor.py` — flujo de investigación actual, single-thread.
- Hardware: RTX 4050 (6GB VRAM efectiva), Qwen3.5-9B EXL3 3.0bpw.

## Notas

### Qué son subagents y por qué valen

Un subagent es un agente con su propio contexto, goal y loop de tools, corriendo en paralelo a otros subagents bajo un orquestador. El patrón clásico:

```
Orquestador (planner)
  ├── Subagent A: "explora ángulo histórico"
  │     └── search → read → summarize → report A
  ├── Subagent B: "explora ángulo técnico"
  │     └── search → read → summarize → report B
  └── Subagent C: "verifica claims de A y B"
        └── search → cross-check → report C

Orquestador sintetiza A + B + C → respuesta final
```

**Valor real para IPA:**

- **Diversidad de ángulos**: una investigación sobre "fotónica" beneficia de un subagent en física, otro en ingeniería, otro en aplicaciones. Hoy un solo hilo tiende a un ángulo dominante.
- **Verificación cruzada**: un subagent puede verificar los claims de otro, reduciendo alucinación.
- **Throughput**: 3 subagents en paralelo hacen 3x trabajo en el mismo wall-clock (si hay VRAM para los 3).
- **Aislamiento de fallos**: si un subagent entra en loop o alucina, no contamina a los otros.

### Por qué NO se implementa ahora

**VRAM es el cuello de botella absoluto.**

El modelo estrella Qwen3.5-9B EXL3 3.0bpw ocupa ~3.5GB de VRAM. La RTX 4050 tiene 6GB. Eso deja ~2.5GB para KV cache + activations. Un solo modelo ya está ajustado:

```
Modelo 9B 3.0bpw:        ~3.5 GB VRAM
KV cache (8K context):   ~1.5 GB
Activations + overhead:  ~0.8 GB
─────────────────────────────────
Total:                   ~5.8 GB  ← queda ~0.2 GB libre
```

**Cargar un segundo modelo 9B es físicamente imposible** en la 4050. Las opciones son:

| Opción | VRAM | Viable? | Calidad |
|---|---|---|---|
| 2× 9B 3.0bpw en paralelo | ~7 GB | **no** — OOM | — |
| 2× 9B 2.0bpw en paralelo | ~5 GB | apenas — KV cache insuficiente | degradada |
| 1× 9B + 1× 3B (Qwen 3B) | ~5 GB | sí, pero el 3B es muy débil | mixta |
| 1× 9B secuencial (time-slice) | ~3.5 GB | sí — pero no es paralelo | igual |
| 1× 9B + CPU model (llama.cpp) | 0 GPU extra | sí — pero lento | mixta |

**Time-slicing** (un modelo, múltiples contextos en cola) es técnicamente posible con continuous batching del exl3_provider, pero:
- No es paralelismo real — es concurrencia con contención.
- El throughput total no sube; de hecho baja por context switching.
- El valor de "diversidad de ángulos" se pierde porque los subagents no exploran simultáneamente.

### Lo que SÍ se puede hacer sin VRAM extra

1. **Subagents determinísticos (no-LLM)**: un subagent que solo hace `search_corpus` + filtrado heurístico no necesita LLM. Puede correr en CPU en paralelo. Útil para "recolección paralela de candidatos" antes de que el LLM juzgue.

2. **Subagents secuenciales con context compartido**: en vez de 3 modelos en paralelo, 1 modelo que hace 3 roles en secuencia, cada uno con un sub-context aislado. Pierdes paralelismo pero ganas diversidad de ángulos. El orquestador (determinístico) secuenciaría.

3. **Subagents async con un solo modelo**: lanzar 3 research_tasks al `research_executor` (que ya es async), cada una con un ángulo distinto. El LLM judge se invoca secuencialmente cuando cada tarea llega al punto de juicio, pero la recolección (web search + scrape) es paralela. Esto **ya es posible** con la infra actual — solo falta el orquestador que lance múltiples `research_topic` con queries derivadas.

## Comparativa

| Patrón | VRAM | Paralelismo real | Diversidad | Implementable hoy? |
|---|---|---|---|---|
| Single-agent (actual) | 1 modelo | no | baja | sí |
| Subagents LLM paralelos | N modelos | sí | alta | **no** (OOM) |
| Subagents determinísticos paralelos | 0 GPU | sí (CPU) | media | sí |
| Subagents secuenciales (1 modelo, 3 roles) | 1 modelo | no | alta | sí |
| Multi-research async (1 modelo, recolección paralela) | 1 modelo | parcial (I/O) | media | sí |

## Takeaways

1. **Subagents LLM paralelos no son viables en la 4050.** No hay VRAM para 2+ modelos 9B. Esta es una restricción física, no arquitectónica.

2. **El valor de subagents (diversidad, verificación) se puede obtener parcialmente sin paralelismo LLM**:
   - Diversidad: orquestador secuencial con roles distintos (1 modelo, 3 pasadas).
   - Verificación: subagent determinístico que cross-checka claims contra el corpus (sin LLM).
   - Recolección paralela: múltiples `research_topic` async (I/O paralelo, LLM secuencial).

3. **No implementar subagents ahora.** La complejidad de orquestación no se justifica sin VRAM para paralelismo real. El costo/beneficio es negativo: mucha complejidad nueva por diversidad marginal.

4. **Re-evaluar cuando cambie el hardware.** Si se agrega una segunda GPU o se migra a un modelo que deje VRAM libre (ej: 3B cuantizado como subagent + 9B como orquestador), el patrón se vuelve viable. Documentar este research como pre-requisito para esa decisión futura.

5. **La arquitectura actual (single-agent + bounded loop + async research) es correcta para el hardware actual.** No forzar un patrón multi-agent que el hardware no sostiene.

## Gaps

- Falta benchmark de VRAM real con 2 modelos cargados (confirmar OOM empíricamente).
- Falta definir el contrato de `SubagentTask` (goal, context_isolated, report_back) para cuando se implemente.
- Falta decidir si el orquestador secuencial (opción 2) vale la pena vs el planner + task queue de RES-005.
- Re-evaluar si un modelo 3B (Qwen-3B, Phi-3-mini) como subagent verificador deja VRAM suficiente para el 9B orquestador.

## Cierre (2026-09-23)

Research concluido con recomendación negativa adoptada: la propuesta de
deliberación multi-agente quedó formalmente rechazada en **DEC-009** (daño
neto medido en EXP-004 + imposibilidad de VRAM documentada aquí). La
arquitectura vigente (single-agent + bounded loop + research async) es la
recomendada por este research. Los gaps restantes son triggers de
re-evaluación ante cambio de hardware/modelo, no trabajo pendiente.
