---
id: DEC-010
category: decision
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [agentic_runtime, fast_path, ingestion, dashboard, promotion_executor, enrichment]
tags: [tier0, tier1, tier2, idle-scheduler, lease, heavy-lock, dirty-flag, orchestration]
related: [PAT-007, PAT-008, DEC-003, DEC-007, PM-004, EXP-008]
supersedes: null
superseded_by: null
author_model: swe-2
affects: ["src/ipa/agentic/**", "outputs/agent/*.lock"]
---

# DEC-010 — Orquestación del corpus en tres tiers con exclusión de Tier 0

## Contexto

El idle scheduler (T1 determinista / T2 con LLM) evaluaba, clusterizaba y
promovía el corpus **mientras el fast path seguía escribiéndolo**. El caso
detonante: un watcher de `run_fast_path` quedó huérfano al reiniciarse el
dashboard (no figuraba en `JOBS` del proceso nuevo, que es el único chequeo
que `_idle()` hacía) y T1 arrancó contra un staging parcial — contención de
DBs, estado inconsistente y un `pipeline_progress.json` congelado en
"running" que nadie cerraba. Además `research_ingest` (ingesta interactiva)
escribía el corpus canónico sin que el scheduler lo considerara trabajo
activo.

La cuenta de idle tampoco tenía semántica correcta: se medía desde la última
actividad *visible para el dashboard*, no desde el fin real de la escritura
del corpus.

## Decisión

Tres tiers con una única regla de exclusión: **nada de Tier 1/Tier 2 mientras
Tier 0 está activo, y la cuenta de idle empieza solo cuando Tier 0 se
libera**.

### Tier 0 — ingesta (prioridad absoluta, excluyente)

Todo lo que escribe documentos/chunks en un corpus:

- **Batch**: `run_fast_path.py` clama el lease cross-process `tier0.lock`
  bajo `outputs/agent/` (formato `pid|owner|ts`, heartbeat 15 s, TTL 300 s,
  robo por `pid_alive` — PAT-007; es un artefacto de runtime, solo existe
  mientras el Tier 0 corre) al arrancar y lo sostiene durante ingesta
  inicial + watch loop + drain final de embeddings. Una segunda instancia
  ve el holder vivo y sale sola (la ingesta es idempotente).
- **Interactiva**: `research_ingest` / `ingest_reviewed_doc` corren bajo
  `heavy.lock` con prioridad interactiva — mismo criterio semántico,
  verificado por `_idle()` vía `heavy_lock.holder()`.
- Tier 0 además emite las señales derivadas en la misma corrida
  (provenance, `normalized_hash`, `published_at`, `duplicate_of_main`,
  `novelty_hint` post-drain, dirty flags) — PAT-008.

### El gate — `_idle()` devuelve no-idle si cualquiera es cierto

| Condición | Cubre |
|---|---|
| `Idle T1/T2` toggle OFF (`idle_enabled.json`) | kill-switch del sidebar |
| `tier0.active()` | lease Tier 0 vivo (incluye watchers huérfanos) |
| `heavy_lock.holder() is not None` | ingesta interactiva en curso |
| `embedding_maintenance.job_active(...)` | drain/lease de índices ajeno |
| `CHAT_BUSY` | usuario conversando |
| algún `proc` vivo en `JOBS` | hijos del dashboard actual |
| pipeline o reporter `status == "running"` | jobs registrados en estado |

Cada iteración no-idle resetea `LAST_ACTIVITY` → **el cronómetro de idle
arranca de cero cuando Tier 0 suelta el lease**, no antes. Además, por ciclo:
`claim_job("idle_scheduler")` (lease de mantenimiento de índices) +
`ENRICHMENT_LOCK` (una instancia de scheduler por store).

### Tier 1 — enriquecimiento determinista (sin LLM, CPU/IO)

Corre en cada tick de 60 s mientras `_idle()` sea true; cada tarea declara
recursos (locks nombrados, orden alfabético → sin deadlocks) y cooldown.
Orden: higiene → consolidación de memoria → topificación (main + reporter,
serializadas por `cluster_store`) → cola de promoción → cognición
determinista → `index_audit` (read-only, último: ve los stores ya
estables; además salta por su cuenta si `tier0.active()`).

T1 es incremental: gate `dirty:<corpus>` + conteo/cobertura/provenance, y
filter-first por ids antes de tocar texto o embeddings.

### Tier 2 — enriquecimiento con LLM (serial, preemptible)

Dos disparadores, motores distintos (split de EXP-008):

- **Pase profundo**: `idle ≥ IPA_IDLE_DEEP_THRESHOLD_MINUTES` (30) y
  `IPA_IDLE_DEEP_ENRICHMENT=1` → carga su propio motor ExL3 batch
  (`_t2_owned`), bajando Ollama si quedó warm del chat.
- **Modelo ya cargado**: `idle ≥ IPA_IDLE_LLM_LOADED_THRESHOLD_MINUTES`
  (5) y `IPA_IDLE_LLM_LOADED_ENRICHMENT=1` → reuso del provider del chat,
  sin cargar nada.

Corre una vez por ventana de idle (`level2_done`/`deep_done` se resetean al
volver a no-idle) y aborta entre items vía `should_abort → _idle()` — un
chat a mitad de pase descarga el motor batch y monta el interactivo.

## Consecuencias

- **Gana**: T1/T2 nunca evalúan estados parciales del corpus; la ventana de
  idle es semánticamente correcta (post-Tier 0); huérfanos de dashboard
  cubiertos por PID-liveness; prioridad interactiva real.
- **Coste**: los writers deben participar del contrato de leases (advisory);
  una corrida Tier 0 larga posterga todo el enriquecimiento — deseado.
- **Reversibilidad**: cada gate es un chequeo aislado en `_idle()`; quitar
  uno no rompe el resto.

## Evidencia

- `src/ipa/agentic/tier0.py` — lease + heartbeat + staleness.
- `src/ipa/ingestion/fast_path_cli.py` — claim/heartbeat/release en `main()`.
- `src/ipa/dashboard/server.py` — `_idle()` (gates) + loop de 60 s +
  disparadores Tier 2.
- `tests/test_tier0.py` — claim, heartbeat, stale steal, exit de duplicado.
- `docs/plans/tier0-signals-idle-optimization.md` — señales y gates de T1.
- Incidente origen: watcher huérfano + `pipeline_progress` congelado
  (2026-09-23); relacionado con PM-004 (starvation de jobs pesados).

## Alcance

Cambia el contrato de activación del enriquecimiento, no las fronteras de
datos: DocumentStore sigue canónico, índices derivados, Landing → Transit →
Archive intacto (DEC-007). Supersede implícitamente la noción de "idle =
sin jobs del dashboard" por "idle = sin Tier 0 ni trabajo pesado".
