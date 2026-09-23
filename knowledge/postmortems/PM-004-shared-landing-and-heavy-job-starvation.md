---
id: PM-004
category: postmortem
status: accepted
created: 2026-09-22
updated: 2026-09-23
author: human
components: [agentic_runtime, research_executor, ingestion, landing_zone, fast_path, embeddings, dashboard]
tags: [starvation, landing, shared-workdir, budgets, cpu, bge-m3, heavy-lock, scheduling, priority, fp8]
related: [PAT-003, PAT-004, DEC-007, EXP-008, EXP-009]
supersedes: null
superseded_by: null
affects: ["outputs/agent/heavy.lock", "src/ipa/agentic/**"]
---

# PM-004 — Research starved by concurrent ingestion: shared landing dir, unbounded post-scrape work, no heavy-job serialization

## Impacto

El usuario pidió por chat una investigación (`research_topic`, presupuesto
`max_seconds=120`) y **después** lanzó una ingesta masiva (pipeline
scraper → fast_path) esperando recibir primero la respuesta de la research.

Lo que pasó en su lugar:

- la research llevaba **27+ minutos** corriendo con un presupuesto declarado de
  120 s (`research_progress.json`: `status: running`, `started_at
  2026-09-22T22:15:59Z`);
- el pipeline competía con ella por los 12 threads de la máquina
  (`run_research.py` acumuló 6661 s de CPU; el fast path 5846 s);
- el dashboard quedó sin refrescar métricas durante minutos (el endpoint
  `/api/state` no llegaba a responder a tiempo), lo que se percibió como "la UI
  está colgada";
- el usuario no tenía ninguna señal de por qué su primera tarea no avanzaba.
- al reiniciar el dashboard durante una suite de tests, se perdió también el
  puntero `_ACTIVE_REPORTER_OUTPUT` in-memory. La UI cayó al output con el
  `report.json` más reciente (vacío) y mostró 0 docs/chunks aunque el corpus
  activo conservaba 127,913 chunks en DocumentStore y miles de vectores en
  LanceDB. Los datos seguían intactos; era una referencia de UI perdida.

## Línea de tiempo

1. `19:15:59` — el chat lanza `run_research.py "Jev sistema arquitectura LLM"
   10 120` (subprocess detached, `research_progress.json` en `running`).
2. `19:19:42` — se lanza la ingesta masiva: `run_web_scrape.py` (599 archivos
   hacia `Landing/web`) + `run_fast_path.py --watch --idle-gate
   .scraper_done` (drain de embeddings en background).
3. La research scrapea a `Landing/web` (dir compartido con el scraper del
   pipeline) y luego ejecuta `ingest_directory("Landing/web")` — ingesta **todo
   el directorio**, no solo sus ~10 documentos aceptados: hereda los ~600
   archivos de la ingesta masiva.
4. `22:21:30` — la LanceDB de staging del pipeline se crea vacía
   (`No existing dataset ... it will be created`) → el drain tiene que embedear
   ~70k chunks del store completo.
5. `22:21:58` — gate de VRAM: `[embed] VRAM libre 397 MiB < 2048 — BGE-M3 en
   CPU`. Con el 9B cargado en 6 GB no hay headroom; BGE-M3 corre en CPU y cada
   batch de 64 chunks tarda ~40 s (`Inference Embeddings ... 39.60s/it`).
6. `22:32` — el usuario observa métricas sin avanzar; las verificaciones de
   estado externas coinciden con la liberación de CPU de los jobs, no con
   ninguna acción sobre el dashboard (el endpoint `/api/health` no tiene
   efectos: solo devuelve un timestamp).

## Causa raíz

Tres defectos que se multiplican:

1. **Dir de trabajo compartido**: `execute_research(landing_dir="Landing/web")`
   hacía que (a) la research ingiriera el directorio entero —violando su propio
   contrato "accepted material only" y heredando la carga de la ingesta masiva—
   y (b) ambos procesos escribieran el mismo árbol.
2. **Trabajo post-scrape sin budget**: `max_seconds` solo se chequeaba dentro
   del loop de scrape. La ingesta y, sobre todo, `_embed_new_chunks()`
   (todo chunk pendiente del corpus canónico, en **una sola llamada** sin
   deadline ni batching) no tenían corte alguno.
3. **Sin serialización de trabajos pesados**: cada tool lanza su subprocess
   detached; no hay cola ni prioridad entre una research interactiva y un
   pipeline de background. El drain del fast path tampoco cedía: la primera
   pasada recorría el corpus completo, así que `stats["indexed"]` quedaba en 0
   (el watch reportaba `0 chunks embedded (draining)`) y el progreso era
   invisible.
4. **Puntero del dashboard solo en memoria**: `_ACTIVE_REPORTER_OUTPUT` se
   perdía ante un restart del watchdog. El fallback elegía el output con
   `report.json` más reciente aunque su corpus estuviera vacío, ignorando el
   corpus realmente poblado que todavía no tenía reporte.

Factor contribuyente: el gate de VRAM (correcto por EXP-008 §10 — evita el
freeze del compositor de Windows) garantiza BGE-M3 en CPU mientras el 9B ocupa
la GPU, así que cualquier trabajo de embeddings compite por CPU, no por GPU.

## Corrección

1. **Dir de trabajo por corrida** (`research_executor._research_run_dir`):
   `landing_dir=None` → `outputs/agent/research/<stamp>-<slug>/`. La ingesta
   recorre solo el material aceptado de esa corrida. El handoff al corpus
   principal no cambia: la ingesta registra `provenance=agent_research` en el
   corpus canónico y `web_source` sigue etiquetando el material web (PAT-003).
   El dir privado queda como traza de auditoría del material fuente. Los otros
   tres consumidores del flujo se alinearon en el mismo cambio: Tutor
   (`execute_approved_research`), CLI (`agent.py research --landing`) y los
   workers de review de docs rechazados (`outputs/agent/research/review/`).
2. **Budget post-scrape** (`max_ingest_seconds`, env
   `IPA_RESEARCH_INGEST_BUDGET`, default 600 s): se chequea antes de ingesta y
   embeddings; `_embed_new_chunks()` ahora **batchea** (64) y corta entre
   batches por deadline, y embebe **solo los documentos de esta corrida**
   (`document_ids`).
3. **Lock de trabajos pesados** (`ipa/agentic/heavy_lock.py`):
   `outputs/agent/heavy.lock` serializa las fases pesadas con prioridad
   interactiva (research, 10) sobre background (pipeline/drain, 50). El drain
   del fast path toma el lock por pasada —ahora acotada a
   `IPA_EMBED_PASS_CHUNKS` (256) chunks, lo que además hace visible el
   progreso— y **cede** (`should_yield`) mientras haya un waiter de mayor
   prioridad registrado. La research reporta la espera
   (`research_progress.json.heavy_wait`) y el dashboard la muestra.
   Advisory y best-effort (PAT-004): si la research no consigue el lock en
   `IPA_HEAVY_LOCK_WAIT` (300 s), sigue igual — nunca se deadlockea una
   respuesta al usuario.
4. **Puntero del output activo durable**: `run_full_pipeline` escribe
   `outputs/web_dashboard/active_reporter_output.json` con replace atómico;
   en el boot, `active_reporter_output()` lo restaura si el directorio sigue
   bajo `outputs/reporter/`. Si falta o es stale, el fallback evalúa todos los
   corpora (incluyendo los que aún no tienen `report.json`) y prioriza los que
   tienen documentos vivos. Tests de regresión cubren restart, stale y output
   reciente vacío.

## Consecuencias aceptadas

- El material fuente de la research ya no entra al ciclo Landing → Transit →
  Archive: vive en `outputs/agent/research/<run_id>/` como traza de auditoría
  (el sweep lo ignora: su `source_uri` no está bajo `Landing/`, y el material
  ya está en el corpus canónico con `provenance=agent_research`). La retención
  de esos dirs es **manual** — no hay poda automática (borrar es una operación
  destructiva que requiere decisión del usuario).
- Un embed cortado por el budget deja chunks pendientes en el corpus: BM25 los
  cubre y el drain/idle los completa después.
- Mientras una research tiene el lock, el drain del pipeline cede: el watch del
  pipeline tarda más en alcanzar su idle-exit. Es el orden pedido
  (responder primero, ingesta masiva después).

## Embeddings: revalidación pareada y modo GPU masivo (2026-09-22)

La estimación inicial de ~125x mezclaba texto GPU sintético corto con chunks CPU
reales y no era válida como comparación de dataset. Repetición sobre los mismos
64 chunks reales del DocumentStore (~512 caracteres, 159 tokens promedio),
`dense+sparse`, misma versión FlagEmbedding/PyTorch y warmup:

| Hardware/config probada | Throughput | Observación |
|---|---:|---|
| CPU FP32, 6 threads, batch 4 | 2.86–2.99 chunks/s | Mejor CPU probada en este host |
| GPU FP16, batch 4 | 123.8–135 chunks/s | GPU óptimo pareado aproximado |
| GPU FP16, batch 8 | 126.6–129.1 chunks/s | Prácticamente igual al batch 4 |
| GPU FP16, batch 16 | 109.2 chunks/s | Menor |
| GPU FP16, batch 64 | 53–54 chunks/s | Padding por longitudes variables reduce el throughput |
| GPU FP32, batch 8 | 36.8–37.1 chunks/s | Igual precisión que CPU; ~14x vs CPU |

La configuración operacional CPU FP32 vs GPU FP16 da ~45x con el mejor batch
medido, no 125x. La brecha sigue siendo grande: FP16 aporta ~3.5x respecto a
GPU FP32 y el resto es aceleración GPU vs CPU en esta carga. Sonda local: misma
muestra de 64 chunks, `embed_texts_hybrid`; CPU batch sweep 4/8/16/64 y GPU sweep
4/8/16 con varias repeticiones. Es microbenchmark de dispositivo, no garantía
de throughput de extremo a extremo.

La cifra CPU de 6 threads corresponde al probe que llamó explícitamente
`torch.set_num_threads(6)`. `EmbeddingAdapter` no fija el thread count: este venv
reporta 6 como default de PyTorch, pero no es un contrato reproducible entre
launchers. El fast path/drain usa el batch interno 4 por default; callers con
override explícito no lo heredan: `continuous_pipeline` defaulta a batch 256,
`lancedb_incremental`/Tier 2 re-embed usan 192 y el semantic chunker 16. Esos
paths no están cubiertos por el microbenchmark de batch 4. `IPA_EMBED_GPU_BULK=0`
desactiva el lease GPU bulk, no fija CPU;
`IPA_EMBED_DEVICE=cpu` es el pin efectivo.

El diseño de FP8 E4M3/NVFP8 para BGE-M3 en Ada está registrado en EXP-009 como
propuesta. No hay integración, prueba ni benchmark FP8 en esta revisión.

Corrección implementada: si el backlog es >=512 (`IPA_EMBED_GPU_MIN_BACKLOG`,
default), fast-path/runner toman un lease exclusivo de VRAM, descargan Ollama,
cargan BGE-M3 FP16 batch 4 y lo mantienen en GPU hasta que LanceDB no tenga
pendientes. Chat se bloquea con mensaje visible (HTTP 423 + banner UI) desde la
preparación hasta el warmup de Ollama al terminar. El provider Ollama mantiene
`vram.lock` durante cada stream; el lote espera una generación en vuelo antes
de descargar. Heartbeat renueva el TTL del lock; el progreso vive en
`embedding_maintenance.json`; LanceDB `chunk_id`s son el checkpoint de resume.
Cancelación graceful entre batches libera VRAM y restaura chat; un crash libera
el lock al detectar PID muerto y la corrida retoma los vectores ya escritos.

La promoción es otro escritor del mismo staging/main pair, así que se serializa
también con un job lock cross-process compartido: el embedding drain toma el
lease durante todo el proceso; el idle scheduler no inicia T1/T2 ni las rutas
manuales de promoción/review/reindex pueden mutar los corpora mientras ese lock
está activo. La promoción T1 que ya estaba en vuelo antes del primer guard
terminó, pero no se repite mientras el drain esté activo.

**Preflight de cobertura vectorial (cierre del incidente):** la purga es la
única operación destructiva de una promoción, y ahora está condicionada —
`promote_documents_to_main` calcula los chunks vivos del source del batch y
verifica que cada uno tenga vector en main LanceDB ANTES de purgar. Si falta
aunque sea uno, la promoción se **difiere**: el source queda intacto, la cola
queda `pending` (no se marca done) y el drain completa los vectores; el ciclo
siguiente reintenta (la copia es idempotente). Si main LanceDB no se puede
leer, también difiere (conservador). Escape de emergencia:
`IPA_PROMOTION_REQUIRE_VECTORS=0` restaura el comportamiento anterior.

El break-even teórico pareado: ahorro por chunk ≈ 1/2.9 - 1/129 = 0.327 s. Con
~40 s para descargar Ollama, cargar BGE y restaurar el chat, ≈122 chunks; si una
ventana obliga además a recargar BGE en CPU (~38 s), ≈240. El umbral adoptado
512 deja margen; para una masa de 124,713 pendientes, la extrapolación de la
sonda es ~16 min en GPU frente a ~11.6 h en CPU (no es una SLA end-to-end).

### Promoción concurrente observada durante la primera corrida

La UI mostró Transit 1,261 → 557 mientras BM25/LanceDB se indexaban. Evidencia
local del log idle: a las `22:33:38` la tarea `promotion_queue` terminó con
`promoted_docs=704`, `promoted_chunks=57,781`, `archived=704`; ese T1 había
empezado antes de deshabilitar idle. Los conteos posteriores confirmaron el
movimiento (no pérdida):

- Reporter staging: 1,491 → 787 documentos vivos; 127,913 → 70,132 chunks;
- main E12: 146,362 → 204,143 chunks;
- la fase Lance de esa promoción copió 5,583 vectores disponibles en ese
  instante. Por eso main quedó con 52,198 chunks sin vector; el batch GPU debe
  completar tanto el staging restante como ese backfill de main.

Se apagó la perilla idle temporalmente (estaba ON) para impedir más promociones
mientras se terminan ambos drains. El estado previo se restaurará al final. No
se reingesta Transit ni se borra material: solo se completan vectores faltantes
en los DocumentStores ya indexados.

### Consolas Windows

El lote inicial usaba `python.exe` en primer plano y las renovaciones de
`vram.lock` lanzaban `tasklist`, produciendo ventanas de consola. La liveness
ahora usa Win32 `OpenProcess/GetExitCodeProcess` sin subprocess; los probes
`nvidia-smi` suprimen consola y `run_embed_drain.py --background` relanza con
`pythonw.exe`, stdout/err a `outputs/web_dashboard/logs/embed_drain.log`. Los
reinicios conservaron los vectores confirmados vía checkpoint `chunk_id`.

## Prevención

- `tests/test_heavy_lock.py` (12 casos): contención, robo de locks stale
  (pid muerto / TTL), re-entrada, registro de waiters, cesión del background y
  espera acotada de `heavy_phase`.
- `tests/test_research_executor.py`: dir privado por default, budget auditado,
  y el embed acotado a los documentos de la corrida (con deadline y batching).
- `tests/test_fast_path.py`: la pasada del drain respeta `max_chunks` y el
  drain cede ante un waiter interactivo y retoma al liberarse.
- `tests/test_promotion_executor.py` (8 casos): la promoción difiere sin
  vectores, completa con cobertura, reintenta tras el backfill (end-to-end con
  la cola), la cola deja el batch diferido como `pending`, el opt-out de
  emergencia y main ilegible → defer (nunca marca done).
- Regla operativa: un trabajo pesado nuevo (reporter, T2 idle) debe declarar su
  fase con `heavy_lock.heavy_phase` y elegir prioridad; el default background
  cede a lo interactivo.
- El presupuesto declarado de una tool debe cubrir **todas** sus etapas, no
  solo la primera: `budget_used` ahora registra `ingest` (deadline, lock,
  excedido).

## Lección reutilizable

Un presupuesto que solo cubre la primera etapa de una tool no es un
presupuesto. Y un directorio de trabajo compartido entre jobs convierte
"ingesta del material aceptado" en "ingesta de todo lo que haya en la carpeta":
el aislamiento del work dir es parte del contrato, no una optimización. Cuando
dos trabajos pesados pueden coincidir, la serialización con prioridad explícita
(interactivo > background) es preferible a confiar en que la CPU alcance.
