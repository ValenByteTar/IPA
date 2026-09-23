---
id: PAT-007
category: pattern
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [agentic_runtime, providers, ingestion, dashboard, operations]
tags: [lease, lock, heartbeat, ttl, vram, heavy-lock, tier0, scheduling, priority]
related: [PM-004, EXP-008, EXP-005, DEC-010, PAT-009]
supersedes: null
superseded_by: null
affects: ["outputs/agent/*.lock", "src/ipa/agentic/tier0.py", "tools/work_permits.py", ".devin/hooks.v1.json"]
evidence: ["src/ipa/agentic/tier0.py"]
author_model: swe-2
---

# PAT-007 — Leases de archivo con heartbeat para serialización de recursos

## Problema

En una máquina con 6 GB de VRAM y varios procesos (dashboard watchdog, fast
path, research interactiva, idle scheduler), dos trabajos pesados
concurrentes producen OOM (ExL3↔Ollama, EXP-008 §9), starvation
(interactivo vs pipeline, PM-004) o evaluación de estados parciales del
corpus. Se necesita serialización cross-process sin un broker.

## Solución

Lease de archivo `outputs/agent/<name>.lock` con formato `pid|owner|ts`:

- **Claim + heartbeat**: el holder renueva el timestamp cada N segundos
  (tier0: 15 s heartbeat, TTL 300 s).
- **Recovery**: un lease se roba si el PID está muerto (check `pid_alive` por
  Win32 `OpenProcess`, sin `tasklist` para no lanzar consolas) o si expira el
  TTL — un kill duro no deja el recurso tomado.
- **Prioridad explícita**: interactivo (research, prio 10) > background
  (pipeline/drain, prio 50); el holder de menor prioridad cede
  (`should_yield`) cuando hay un waiter registrado.
- **Advisory, no mutex**: quien no respeta el lock no se bloquea — pero todos
  los writers relevantes lo chequean antes de la fase pesada.

Locks vigentes:

| Lock | Serializa | Holder típico |
|---|---|---|
| `vram.lock` | GPU entre motores | ExL3 `load()`; Ollama lo respeta por stream |
| `heavy.lock` | fases pesadas del corpus | research interactiva / drain fast path |
| `tier0.lock` | ingesta fast_path entera | `run_fast_path.py` (bloquea T1/T2 idle) |
| job lock embed | drain vs promotion/reindex | bulk GPU / rutas manuales |

## Trade-offs

- Advisory: protege solo contra writers que participan; un proceso ajeno al
  contrato puede pisar el recurso.
- TTL alto → recuperación lenta tras crash; TTL bajo → riesgo de robo con un
  heartbeat perdido (Windows ocupado). 300 s con heartbeat 15 s es holgado.

## Ejemplos locales

- `src/ipa/agentic/heavy_lock.py`, `src/ipa/agentic/tier0.py`,
  `ipa/providers/exl3_provider.py` (`vram.lock`), job lock del embed drain.
- Tests: `tests/test_heavy_lock.py` (12 casos: contención, stale steal,
  re-entrada, cesión, espera acotada).
