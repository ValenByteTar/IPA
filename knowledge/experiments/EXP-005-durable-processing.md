---
id: EXP-005
category: experiment
status: accepted
created: 2026-09-06
updated: 2026-09-23
author: human
components: [orchestration, observability, configuration]
tags: [e8, durable-processing, jobspec, jobrunner, process-state, recovery, backpressure, retry]
related: [PM-001, PAT-002]
supersedes: null
superseded_by: null
affects: ["src/ipa/ingestion/**"]
---

# EXP-005 — Procesamiento durable: JobSpec/JobRunner/process_state (E8)

## Hipótesis

La infraestructura de procesos (`ipa.dashboard.process_specs` / `process_runner` / `process_state`) cumple los invariantes de Stage 4 (completion, fallo, stuck, cancelación, recovery, retry, backpressure, estado durable) en un entorno aislado y reproducible.

## Configuración

- Worker sintético generado en directorio temporal (sin GPU, sin LLM, sin escrituras de producción).
- 8 escenarios sobre el `JobRunner` real: completion, propagación de fallo, detección de stuck en vuelo, pause/resume (hammer), recovery tras fallo inyectado, retry acotado con backoff exponencial, backpressure por slots de recurso, durabilidad atómica del estado bajo lecturas concurrentes.
- Estado en directorio aislado; reporte conforme a `contracts/experiment_report.schema.json`.

## Ejecución (2026-09-06, segunda corrida tras implementar mejoras)

- Runner: `src/ipa/dashboard/process_eval.py` + entrypoint `scripts/benchmarks/run_process_eval.py`.
- Artefactos: `outputs/experiments/E8/report.json` (validado con `validate_experiment_report.py`) y `outputs/experiments/E8/scenarios.json` (detalle por escenario, hash referenciado).

## Mejoras implementadas en esta pasada (plan orchestrator-improvements)

1. **Fix de idle**: el runner medía el idle después de refrescar `last_output_time` (idle siempre ~0, stuck inalcanzable). Ahora el idle se mide contra la ventana real de silencio y el runner escribe `idle`/`stuck` en vuelo.
2. **`run_job_with_retry`**: reintentos acotados con backoff exponencial (`--retries`, `--backoff` en el CLI `run_job.py`); el contador de intentos queda en el state.
3. **Backpressure por slots**: `acquire_slot`/`release_slot` en `process_state.py` — concurrencia acotada por recurso con robo de slots stale (`--resource gpu --max-concurrent 1` en el CLI; rechazo con exit 75/EX_TEMPFAIL para re-queue).

## Resultados (8/8 escenarios)

- **completion**: líneas parseadas, métricas acumuladas, estado `done`, exit 0.
- **failure**: exit non-zero propaga a estado `error`.
- **stuck (post-fix)**: una línea que llega tras el umbral de idle escribe `stuck` en vuelo; estado final `done`.
- **pause/resume**: pause file del hammer bloquea (`paused`) y reanuda.
- **recovery**: tras fallo inyectado (exit 2, estado `error`), la re-corrida transiciona `running` → `done`.
- **retry_backoff**: worker que falla el primer intento; `run_job_with_retry` lo recupera en el intento 2 con backoff exponencial; `attempts: 2` registrado en state.
- **backpressure_slots**: con `max_concurrent=1`, la segunda adquisición es rechazada; tras liberar, se re-adquiere.
- **atomic_state**: 40 escrituras atómicas con lecturas concurrentes — ningún lector observó JSON parcial.

## Hallazgos residuales

- `final_status` prioriza la línea de completitud sobre el exit code: un worker que emite DONE y sale con código non-zero se reporta `done`. Los workers reales solo emiten DONE en éxito; semántica a conocer.
- Colas persistentes no implementadas: la backpressure es por slots de recurso; un rechazo (exit 75) debe ser re-encolado por el llamador.

## Kilometraje operacional (2026-09-06, misma pasada)

Jobs reales del orquestador con `--retries`/`--resource`:

1. **pipeline** (`--idle-timeout 15 --retries 2 --resource pipeline --max-concurrent 1`): procesó 1 archivo real de Landing → 159 chunks, exit 0, `attempts: 1`, slot adquirido/liberado, estado `done`.
2. **lancedb** (`--retries 2 --resource gpu --max-concurrent 1`): embebió los 159 chunks nuevos (4.549 total, missing 0, FTS creado), exit 0, `attempts: 1`, fix de Unicode validado con salida real del worker.
3. El primer intento de lancedb **encontró y recuperó un bug real vía el mecanismo de retry**: el `print()` del runner crasheaba con salida Unicode (charmap/cp1252) cuando el stdout padre no es UTF-8. El retry lo registró (`attempts: 2`, `retry_exception`) y el fix (reconfigure del stdout del runner a UTF-8 con replace) lo eliminó.

Corpus actualizado como efecto del kilometraje: E12-corpus pasó de 4.390 a **4.549 chunks** (102 → 103 documentos), Landing vacío, 104 archivos archivados.

## Conclusión

Los invariantes de Stage 4 implementados se verifican en entorno aislado (8/8 escenarios) y con kilometraje operacional real sin incidentes. Nivel de promoción: **`preferred`** — para `production` queda el uso sostenido del orquestador en operación normal del sistema.

## Recomendación

- [x] Mantener como Experiment
- [ ] Congelar como Benchmark
- [ ] Crear Decision
- [ ] Proponer ADR
- [ ] Nothing
