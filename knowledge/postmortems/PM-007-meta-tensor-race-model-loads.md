---
id: PM-007
category: postmortem
status: accepted
created: 2026-09-24
updated: 2026-09-24
author: agent
components: [indexes, providers, dashboard, retrieval, acquisition]
tags: [meta-tensor, accelerate, concurrency, warmup, monkey-patch, sse-masking, retrieval]
related: [PAT-007, EXP-008, PM-004]
supersedes: null
superseded_by: null
affects: ["src/ipa/model_load_lock.py", "src/ipa/indexes/embedding_adapter.py", "src/ipa/indexes/reranker_adapter.py", "src/ipa/acquisition/ocr_adapter.py", "src/ipa/providers/exl3_provider.py", "src/ipa/dashboard/server.py", "src/ipa/dashboard/api.py", "web/static/app.js"]
evidence: ["src/ipa/model_load_lock.py", "tests/test_model_load_lock.py", "src/ipa/dashboard/api.py"]
author_model: swe-2
trigger: permit:PW-20260924-01
---

# PM-007 — Race de carga de modelos: params en `meta` + retrieval error enmascarado como corpus vacío

## Impacto

Tras un restart del dashboard (2026-09-24), **todo** retrieval devolvía
`Cannot copy out of meta tensor; no data!` y la UI lo mostraba como "Sin
resultados en el corpus" + botón de investigación manual. El corpus tenía
82 docs / 1522 chunks sobre el tema consultado (Jev) — el dato existía,
el índice estaba sano, la búsqueda en un proceso limpio devolvía 10 hits.
El fallo era **persistente por proceso**: cada query del usuario fallaba
igual durante todo el uptime, disparando investigaciones web redundantes.

## Causa raíz — dos bugs compuestos

1. **Race de construcción de modelos (meta-tensor).** El dashboard
   arranca `_warmup_provider` y `_warmup_embeddings` en threads
   paralelos (`server.py` ~2119-2191). `transformers`/`accelerate`
   (`init_empty_weights`, `init_on_device`) parchean
   `nn.Module.register_parameter` **a nivel clase** durante
   `from_pretrained` — el patch es global al proceso, no thread-local.
   Cualquier `nn.Module` construido por otro thread en esa ventana
   (segundo adapter por la check-then-act race de `get_embedding_adapter`,
   reranker lazy, ExL3 `Model.from_config`, easyocr) registra sus params
   en device `meta` y nunca los materializa. Además, dos contextos
   `init_empty_weights` solapados corrompen el save/restore de
   `register_parameter` (el segundo restaura la versión ya parcheada del
   primero → el patch queda filtrado permanentemente).
2. **SSE enmascaraba el error.** `api.py` emitía `stage: empty` en el
   `else` de `if _auto_hits` — que también corría tras `timeout`/`error`.
   La UI pisaba "Error en retrieval: …" con "Sin resultados en el corpus"
   y el ctx volátil le decía al modelo "decí que no hay datos": una
   falla técnica se presentaba como ausencia de conocimiento.

## Fix

- `src/ipa/model_load_lock.py` (nuevo): `MODEL_LOAD_LOCK` — RLock global
  que serializa toda construcción pesada de modelos in-process:
  `BGEM3FlagModel` (embedding_adapter), `FlagReranker` (reranker_adapter),
  `easyocr.Reader` (ocr_adapter), `_load_locked()` de ExL3 (bajo
  vram.lock → MODEL_LOAD_LOCK, orden fijo). Double-check del singleton
  dentro del lock + tripwire: si el modelo cargado tiene params `meta`,
  `_ensure_model` falla fuerte en vez de dejar un singleton corrupto.
- `get_embedding_adapter` (server.py): lock propio para el singleton —
  sin él, dos threads creaban dos adapters → dos BGE-M3 (~2 GB c/u).
- `api.py`: `error`/`timeout` ya no emiten `empty` encima; el ctx volátil
  dice "la búsqueda falló por error técnico" en vez de "no hay datos".
  El `empty` genuino lleva `auto: true` cuando el auto-research va a
  disparar, y `app.js` lo anuncia ("investigando en la web
  automáticamente…") con el botón como escape hatch.

## Lecciones

- **Todo constructor de `nn.Module` pesado es sección crítica** cuando
  transformers/accelerate convive en el proceso: sus contextos de
  init parchan estado global de clase. PAT-007 (file leases) serializa
  procesos; hacía falta su complemento intra-proceso.
- **Nunca traducir un error a un estado de dominio** ("no hay datos") en
  la capa de presentación: `empty` debe emitirse solo cuando la búsqueda
  corrió y devolvió cero. El enmascaramiento escondió el bug durante
  todo un uptime y generó research web espurio.
- Warmups paralelos ahorran ~30 s de arranque pero multiplican las
  ventanas de race; con el lock el warmup sigue siendo paralelo (los
  loads se serializan, el resto del trabajo no).
- Verificación en vivo: la query que fallaba
  (`Hablame sobre la aquitectura de Jev`) devolvió `found` con 8 hits
  reales tras el fix + restart.

## Tests

`tests/test_model_load_lock.py` — 5 casos: ctor bajo lock (probe desde
thread ajeno), single-construction bajo contención (4 threads → 1 ctor),
tripwire meta (raise + reset del singleton), reranker y OCR bajo lock.
