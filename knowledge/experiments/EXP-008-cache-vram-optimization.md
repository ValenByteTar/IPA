---
id: EXP-008
category: experiment
status: accepted
created: 2026-09-19
updated: 2026-09-19
author: agent
components: [providers, dashboard, indexes, operations]
tags: [kv-cache, prefix-cache, ollama, exl3, vram, num-gpu, num-keep, batch, throughput]
related: [EXP-004, EXP-007]
supersedes: null
superseded_by: null
---

# EXP-008 — Caches y VRAM: PT cache, KV, num_gpu, num_keep y convivencia de motores

## Hipótesis

El rendimiento del chat (tok/s, TTFT) y la estabilidad en contexto largo están
limitados principalmente por (a) cómo se dimensiona y reusa el KV cache, (b) el
reparto de capas GPU/CPU y (c) la contención de VRAM entre los dos motores
(Ollama y ExL3). Cada punto se puede medir por separado con las métricas que ya
expone cada motor.

## Configuración

- **Hardware**: RTX 4050 Laptop, 6141 MiB (6 GB), Windows/WDDM.
- **Ollama 0.34.2** — `qwen3.5:9b-q4_K_M` (6.3 GB), `OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KV_CACHE_TYPE=q8_0` (nuevo).
- **ExLlamaV3** — `Qwen3.5-9B-exl3-3.0bpw` + MTP (`mtp.safetensors`),
  `cache_k/v_bits=8`, batch 1 (interactivo) / 6 (batch).
- **Métricas**: `prompt_eval_count` / `prompt_eval_cached_count` /
  `eval_count` + duraciones (Ollama, por request);
  `time_to_first_token_ms` / `tokens_per_second` (provider ExL3);
  `nvidia-smi` para VRAM física.
- **Scripts**: `scripts/operations/_measure_llm.py`, `_ollama_ab_test.py`,
  `_exl3_fatigue_test.py`, `_exl3_cache_fix_check.py`, `_exl3_prefix_check.py`,
  `_exl3_batch_tuning.py`. Reportes en `outputs/experiments/`.

## Resultados

### 1. Ollama — reparto de capas (`num_gpu`)

El auto-fit de Ollama es conservador: dejaba ~1.4 GB de VRAM libre y mandaba la
mitad del modelo a CPU.

| `num_gpu` | capas GPU | decode | prefill | VRAM libre |
|---|---|---|---|---|
| auto | 18/34 | 10.4 tok/s | 453 tok/s | 1482 MiB |
| 24 | 24/34 | 12.5 tok/s | 588 tok/s | — |
| 28 | 28/34 | 17.3 tok/s | 653 tok/s | 1468 MiB |
| **30** | **30/34** | **19.4-20.6 tok/s** | 680 tok/s | 1228 MiB |
| 32 | 32/34 | 9.3 tok/s | 22 tok/s | 956 MiB |

32 capas colapsa (sin VRAM para los buffers de cómputo). Sweet spot: **30**.
Ganancia: **~2x decode** (10.4 → 20 tok/s), verificado también en el chat real
(18.9 tok/s).

### 2. Ollama — PT cache (prefix reuse)

El runner reusa el longest-common-prefix por slot (checkpoints cada ~500 tok).
El prompt original ponía TODO el contenido volátil (evidencia RAG, memoria,
catálogo de tools) dentro del system prompt → el prefijo compartido moría a los
~500-1100 tokens y el runner **rechazaba el reuse** (similitud < umbral ~0.7).

Cambio: `[system inmutable] + [historia append-only] + [volátil al tail, dentro
del turno user]`.

| Config | similitud | tokens evaluados |
|---|---|---|
| volátil en system (antes) | 0.41-0.47 | 822-4068 (full, reuse rechazado) |
| volátil en tail (después) | 0.74-0.90 | 270-719 (72-90% reuse) |

Métrica directa por request: `prompt_eval_cached_count` (1,621/2,198 = 74% y
2,194/2,389 = 92% en turnos reales; prefill 401 ms).

### 3. Ollama — `num_keep` y `num_ctx`

`n_keep = 4` (default de llama.cpp): al llenarse el contexto, los primeros
tokens — system prompt, identidad, grounding — son lo PRIMERO en descartarse.
Con `num_keep=2048` el prompt de instrucciones sobrevive al context shift.
`num_ctx` 8192 → 6144 (los prompts reales miden 2.3-4.1k) libera ~68 MiB de KV
por slot.

### 4. Ollama — sampler (`rep_p` × `repeat_last_n`)

4 configs × 3 corridas de 300 tok: 19.5-21.2 tok/s, stopwords 0.35-0.38,
**0 drift en todas**. Sin diferencia significativa → defaults sin cambios
(los knobs quedan expuestos por si otra GPU/modelo lo justifica).

### 5. ExL3 — el cache estaba mal dimensionado (bug real)

`cache_tokens = min(context_length, mtp_cache_tokens)` = `min(6144, 4096)` =
**4096**, no los 6144 declarados. Consecuencias medidas:

| Prompt | Resultado |
|---|---|
| 2,656 tok | genera bien |
| 4,376 tok | **0 tokens, en silencio**: `Job requires 19 pages (only 16 available)` |
| ~4,000 tok + 50-300 tok salida | cruza 4096 a mitad de generación → pierde el inicio → **divaga** |

Fix: `max(context_length, mtp_cache_tokens)`. Verificado: el prompt de 4,376
tokens que fallaba ahora genera limpio (0/3 drift, 24 tok/s).

### 6. ExL3 — fatiga (¿se degrada el 9B?)

27 generaciones de 300 tok bajo el cap (prompt corto y contexto hasta 2.6k) en
todas las configs (KV q8/fp16, rep_p 1.15/1.0, no_think on/off, suppress_cjk
on/off): **0 drift** (repetición, gibberish, eco de sección). La atribución
vieja ("el 9B se degrada a los ~80 tokens") era en gran parte el bug de cache
del punto 5. `rep_p=1.0` fue consistentemente más rápido (42.3 vs 37.2 tok/s) y
más natural (stopwords 0.38-0.49 vs 0.24-0.36).

### 7. ExL3 — PT cache: ya existe en la librería

El generator hashea las páginas de KV completas y reusa prefijos compartidos
entre jobs. Medido (mismo prompt, 2190 tok):

```
run 1 (frío): TTFT 3734 ms
run 2:        TTFT  344 ms   ← 10.9x
run 3:        TTFT  344 ms
```

No hay que implementar nada: el provider ya se beneficia.

### 8. ExL3 — batch en 6 GB y el efecto de MTP

- `batch_size=6` + `ctx 6144` → **OOM al cargar** (6 secuencias × KV no entran).
- `batch_size=6` + `ctx 2048` **con MTP** → colapso: 702 tok en 40.2 s = 17.5 tok/s
  agregado, secuencias de **3.1 a 31.3 tok/s** (thrashing).
- `batch_size=6` + `ctx 2048` **sin MTP** → **83.2 tok/s uniforme** (29.0 por
  secuencia, stdev 1.2). **El colapso era MTP, no la presión de KV.**
- **Sweep con y sin MTP** (ctx 2048, 6 prompts × 120 tok, 3 reps, wall time):

| batch | con MTP (tok/s) | sin MTP (tok/s) | wall (sin MTP) | stdev sin MTP |
|---|---|---|---|---|
| 1 | 35.8 | 32.2 | 22.0s | 0.7 |
| 2 | 58.5 | 56.2 | 12.3s | 0.3 |
| 3 | 78.0 | 82.5 | 8.7s | 0.0 |
| 4 | 74.2-81.5 | 75.8-83.2 | 8.6s | 0.9 |
| 5 | 80.4 | — | — | — |
| 6 | **17.5** | **83.2** | 8.6s | 1.2 |

- **El throughput agregado satura a batch ≥3** (~83 tok/s = 714 tok en 8.6 s):
  está limitado por ancho de banda de memoria, no por el grado de batch. El
  sweet spot es el batch más chico que satura: **3-4**.
- **MTP**: +14% a batch 1 (interactivo), +10% a batch 2, neutro a batch 4,
  **catastrófico a batch 6** (17.5 tok/s: peor que no usarlo).

#### Por qué MTP colapsa en batch (mecanismo)

El speculative decoding genera tokens con un draft y los **verifica** en un
paso del modelo target. En batch, cada secuencia tiene una aceptación distinta
(longitudes "ragged"), y tras cada ronda de verificación hay que **realinear**
position IDs, attention masks y estado del KV cache entre las secuencias del
lote. Ese costo de realineación **crece superlinealmente con el batch**:

- arXiv 2510.22876 (*Batch Speculative Decoding Done Right*): mide
  *"realignment consumes up to 40% of computation at batch size 8"* y lo
  describe como *"an inherent cost of algorithmic correctness across ragged
  tensors, not an implementation inefficiency"*.
- arXiv 2310.18813 (*The Synergy of SD and Batching*): *"the optimal
  speculation length varies across batch sizes... larger batch sizes require a
  smaller speculation length, and a speculation length too large will
  deteriorate the performance"*. A batch 4 con speculation 3 todavía miden
  1.93x; a batch 16-32 los speculation lengths altos degradan.
- MLSys 2026 (*SD: Performance or Illusion?*): *"the speedup decreases as the
  batch size increases, consistent with the reduced opportunities for SD on
  computation-bound scenarios"*.

Traducción a este sistema: con 6 secuencias, el trabajo de realineación
(verificaciones rechazadas + re-sincronización) supera el ahorro del draft →
las secuencias se desincronizan y el generador reencola (de ahí las tasas de
3 a 31 tok/s). Sin MTP no hay draft ni realineación: el batch 6 rinde igual que
el 4, **uniforme**.

**Conclusión operativa**: el MTP es una optimización de *latencia en batch 1*
(interactivo). En batch es, en el mejor caso, neutro; en el peor (batch ≥5 con
este modelo), destructivo. El guard lo desactiva para `batch_size > 2`.
- **Guard implementado**: `batch_size > 2` → MTP se desactiva solo (con aviso;
  `IPA_EXL3_FORCE_MTP=1` lo fuerza). `create_star_provider` default 6 → 4.
  Test: `tests/test_device_fallback.py::test_exl3_disables_mtp_for_batch`.

### 9. VRAM — huérfanos y contención entre motores

- Al reiniciar Ollama, los `llama-server.exe` **sobreviven al padre** y retienen
  GB de VRAM: medido 4448 MiB usados con la API caída; 97 MiB tras matarlos. El
  auto-fit siguiente ve esa VRAM ocupada y manda el modelo a CPU. El watchdog
  ahora los barre cuando la API está caída.
- ExL3 + Ollama no conviven: con Ollama recargando su modelo, ExL3 murió con
  `GPU assert: out of memory` en `rep_pen.cu`. Implementado `vram.lock`
  (`ExL3.load()` descarga Ollama y toma el lock; `OllamaProvider.load()` y
  `generate_chat_stream()` lo respetan → error claro en vez de OOM; un lock de
  proceso muerto se roba por TTL). Verificado end-to-end: el chat devuelve
  `"GPU ocupada por exl3 (batch ExL3 en curso)"`.

### 10. VRAM — agotamiento congelaba la UI (hallazgo crítico)

BGE-M3 (~2.2 GB) cargando en CUDA mientras el modelo del chat ocupaba la GPU
dejó 383 MiB libres → el compositor de Windows se quedó sin memoria de video y
**la interfaz completa (Devin Desktop incluida) se congeló**. Además de la
contención, esto se agravó por `num_gpu=30` (punto 1), que subió el consumo del
modelo de ~2.7 a ~4.4 GB.

Fix: tests herméticos respecto de la GPU — `IPA_EMBED_DEVICE=cpu` en
`tests/conftest.py` (mismo criterio que `IPA_RERANK=0`) + limpieza de los caches
de proceso entre tests. Verificado: `test_index_adapters.py` pasa con la VRAM
en 89 MiB (sin tocar la GPU) y la suite completa corre sin congelar la UI.

## Suite de tests

`903 passed, 1 skipped` en 3:49 (antes del fix de hermetismo la suite congelaba
la UI; un fallo intermedio — `list_promotions` servía un hit viejo del cache de
tools — se corrigió limpiando los caches de proceso entre tests).

### 11. Dónde trabaja ExL3: el pase Tier 2 profundo

Mapeo de cargas LLM y su perfil (salida) para decidir el lugar:

| Carga | Trigger | Salida | Veredicto |
|---|---|---|---|
| Chat/Tutor | usuario | prosa 300-800 tok | ❌ switch por turno |
| Tier 2 `_t2_loaded` | modelo warm + 5 min idle | cortos | ❌ ya hay un modelo cargado |
| **Tier 2 `_t2_deep`** | ≥30 min idle | labels + veredictos | ✅ **único punto que YA carga un modelo** |
| `research_review` | 60s idle | 128 tok | ✅ perfil, ⚠️ trigger corto |
| Consolidador / planner / principios | varios | 500-800 tok | ❌ fuera de rango |

Diseño implementado:
- `_t2_deep` carga **ExL3** (`IPA_T2_ENGINE=exl3`, ctx 2048, batch 4, MTP off por
  el guard) con **fallback a Ollama** si ExL3 no carga; `unload` + release del
  lock al terminar.
- **Split por longitud**: `cog_principles_llm` (~500 tok) se saltea en el pase
  ExL3; `deep_topify` (labels) y la nueva tarea `review_batch` (128 tok) corren
  batched.
- **`review_batch`** (Tier 2, prio 15): la cola de review en un pase batched —
  el switch no se amortiza por pocos docs, sí dentro del pase.
- **`enrich_chunks`** (Tier 2, prio 25): recupera el ex-job `enrichment` del
  Orchestrator (era ExL3 4B cargando su propio modelo — en 6 GB no caben dos,
  así que el módulo `ipa.agentic.chunk_enrichment` corre sobre el 9B del
  pase). Summary + 3 queries sintéticas por chunk + re-embed en LanceDB, con
  checkpoint `embedding_status` (pending→complete) que recupera corridas
  cortadas. Última del pase por volumen: las tareas rápidas se completan
  aunque el usuario vuelva a mitad.
  - Dato del corpus real: chunks uniformes de ~512 chars → el `min_chars=800`
    original no seleccionaba nada (0/140k). Con `min_chars=400`
    (`IPA_T2_ENRICH_MIN_CHARS`) la densidad discrimina **4,572 chunks**
    (~3%); incremental `IPA_T2_ENRICH_LIMIT=60`/pase.
- `IPA_IDLE_DEEP_ENRICHMENT` default **0 → 1** (el path es ahora el hogar
  diseñado del trabajo batch).

**Verificación de integración** (`scripts/operations/_t2_review_batch_check.py`):
4 docs → 4 veredictos en **5.8 s = 0.69 docs/s ≈ 88 tok/s agregado** (2.6x el
serial), con el guard de MTP disparando automáticamente. Nota: 1/4 ítems no
emitió JSON válido (se marca `error` y se reintenta en el próximo pase).

## Decisiones

1. `IPA_OLLAMA_NUM_GPU=30` (medido; el punto de quiebre depende de la GPU).
2. Prompt del chat: inmutable en system, volátil al tail.
3. `num_keep=2048`, `num_ctx=6144`.
4. ExL3: cache `max(context_length, mtp_cache_tokens)`.
5. Lock de VRAM entre motores; Ollama solo lo respeta, no lo toma.
6. Tests herméticos: embeddings en CPU, caches limpiados entre tests.
7. Batch ExL3 en 6 GB: contexto ≤2048; **batch 3-4** (satura el ancho de banda;
   más grande no aporta) y **MTP off** en batch (el guard lo desactiva solo
   para batch > 2; a batch 1 el MTP da +14%).
8. Orchestrator de consola deprecado (la cadena scraper→fast_path→lancedb→
   hammer→enrichment); los jobs van por el dashboard y el idle scheduler.

## Reproducción

```powershell
.venv\Scripts\python.exe scripts\operations\_measure_llm.py
.venv\Scripts\python.exe scripts\operations\_ollama_ab_test.py --repeat 3
.venv\Scripts\python.exe scripts\operations\_exl3_fatigue_test.py --repeat 5
.venv\Scripts\python.exe scripts\operations\_exl3_prefix_check.py
.venv\Scripts\python.exe scripts\operations\_exl3_batch_tuning.py
```
