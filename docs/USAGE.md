# IPA — Guía de uso

Instalación, arranque y uso de todo el sistema en una sola página.
Para reglas operativas completas ver `AGENTS.md`; para decisiones
arquitectónicas, `docs/DECISION_LOG.md`.

## Instalación

Requisito: Python 3.12 (`py -3.12` en esta máquina).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,parsers,web,mcp]"
.venv\Scripts\python.exe -m pytest -q
```

- `requirements.txt` — rangos curados por perfil.
- `requirements.lock` — freeze exacto del entorno (reproducibilidad).

### Perfiles opcionales

| Perfil | Contenido |
|---|---|
| (base) | PyYAML, rich, pytest, PyMuPDF, rank-bm25, numpy |
| `retrieval` | tantivy, lancedb, sqlite-vec, sentence-transformers, transformers |
| `parsers` | docling, unstructured[pdf] — **solo benchmark E3**, no en ingesta |
| `web` | requests, trafilatura, playwright, easyocr |
| `tutor` | torch (CUDA), exllamav3, transformers, flash-linear-attention |
| `mcp` | mcp>=1.0 |
| `all` | todo lo anterior |

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval,parsers,web,mcp]"
.venv\Scripts\python.exe -m pytest -q
```

### Dependencias externas

El core funciona sin ninguna de estas (el smoke path no las toca), pero cada
capacidad las necesita:

| Dependencia | Para qué | Instalación |
|---|---|---|
| **Docker Desktop** | SearXNG local = backend real de `research_topic` / auto-research | [docker.com](https://www.docker.com/products/docker-desktop/) — una vez instalado, `start_ipa_dashboard.ps1` y el watchdog lo levantan y mantienen solos (arrancan Docker Desktop si el daemon está caído y corren `docker compose -f .devin/searxng/docker-compose.yml up -d`). Sin Docker, la búsqueda web cae al fallback DDG, que es best-effort y falla seguido por anti-bot. |
| **Ollama** | Backend LLM del chat/Tutor (CPU y fallback sin GPU) | [ollama.com](https://ollama.com) → `ollama pull qwen3.5:9b-q4_K_M` (o el modelo de `IPA_OLLAMA_MODEL`). El launcher auto-arranca `ollama serve` si está instalado. |
| **Playwright browsers** | Scraping de sitios JS (`--engine playwright/auto`) | `.venv\Scripts\python.exe -m playwright install chromium` |
| **ExLlamaV3** | Modelo estrella por GPU (chat/Tutor con VRAM) | Opcional: requiere CUDA + la extensión compilada en `exllamav3-dev/` (sm_89) y pesos propios — no van en el repo. Sin esto, todo corre por Ollama CPU. |

## Arranque

```powershell
# Dashboard web (un solo comando, abre el navegador)
.\start_ipa_dashboard.bat

# Manual
.venv\Scripts\python.exe scripts\operations\web_dashboard.py --host 127.0.0.1 --port 8765

# CLI del agente (misma identidad y memoria que el dashboard)
.venv\Scripts\python.exe scripts\cli\agent.py chat
.venv\Scripts\python.exe scripts\cli\agent.py chat --role tutor --llm -m "mensaje"
```

**DEPRECADO**: el Orchestrator de consola (cadena scraper → fast_path →
lancedb → hammer → enrichment) no recibe nuevas funciones — los jobs se
lanzan desde el dashboard y el trabajo LLM en background corre por el idle
scheduler (Tiers 1/2). Sigue disponible por compatibilidad
(`scripts\operations\orchestrator.py`); su job `enrichment` (ExL3 4B) queda
sin trigger activo hasta que se redefina su reemplazo.

URL local: `http://127.0.0.1:8765` · Health: `/api/health`

## Chat del agente (dashboard)

- **Selector de rol** en el input: `General` / `Tutor`.
- **General**: chat reactivo con protocolo de tools acotado (máx. 3 rondas por
  turno, desbloqueo progresivo). Detecta intención de aprendizaje ("quiero
  aprender X") y ofrece pasar a Tutor.
- **Deep dive**: "Profundizar" desde un reporte abre el chat con contexto del
  corpus del reporte (los endpoints `/api/deep-dive*` standalone están
  deprecados).
- Herramientas principales: `search_corpus`, `research_topic`,
  `compile_report`, `recall_memory`, `plan_task`, `get_user_profile`,
  `list_topics`, `get_system_status`.

## Tutor

1. `"quiero aprender X"` → diagnóstico determinístico sobre corpus + mastery.
2. Corpus insuficiente (<3 conceptos) → propone investigación web → **gate
   humano** (botón en el chat). Mientras corre, el tutor responde de forma
   determinística que aguardamos la fuente web.
3. Roadmap propuesto (LLM) → **Aprobar / Rechazar / Debatir**. Debatir toma
   feedback y re-propone (v+1, supersedes); el gate humano sigue aplicando.
4. Lecciones con avance determinístico ("siguiente unidad", "ya entendí").
5. Assessment estructurado (JSON + abstención) → mastery persistido.

Los roadmaps se pueden archivar (flag operativo, el progreso se conserva).
Las sesiones procesadas por el consolidador automático se marcan `✦ resumida`.

## Aprobaciones (human-in-the-loop)

Cola unificada en el panel Aprobaciones: consolidaciones de memoria,
inferencias de mastery/user model, roadmaps del Tutor, research requests,
curación del Reporter. **Nada se aplica sin aprobación humana** (PAT-004).

## Procesos idle (scheduler)

`ipa/agentic/idle_scheduler.py` orquesta el trabajo en segundo plano con tiers,
prioridades y locks por recurso:

- **Tier 1** (sin VRAM, paralelo): higiene de sesiones → consolidación de
  memoria → topificación (clustering + curación) → promoción → inferencias
  cognitivas (user model, skills, principios, agenda).
- **Tier 2 (LLM)**: re-etiquetado de tópicos, clasificación de grises,
  veredictos de la cola de review (batched), principios abstractos y
  `enrich_chunks` (summary + queries sintéticas por chunk + re-embed —
  recupera el ex-job `enrichment` del Orchestrator sobre el 9B del pase).
  Solo con idle profundo (≥30 min puede cargar el modelo — ExL3 batch por
  defecto) o aprovechando un modelo ya cargado (≥5 min quieto). Preemptible
  al primer mensaje.

Log auditable: `outputs/web_dashboard/logs/idle_enrichment.log`.

## MCP server

Proxy delgado hacia el dashboard — no carga modelos ni implementa retrieval
propio: cada tool es un HTTP call a `/api/tools/execute`. Requiere el
dashboard corriendo (`IPA_PROXY_URL`, default `http://127.0.0.1:8765`).

```json
{
  "mcpServers": {
    "ipa": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["-m", "ipa.mcp.mcp_server"],
      "env": { "PYTHONPATH": "C:\\path\\to\\IPA\\src" }
    }
  }
}
```

Superficie (generada del catálogo del registry al arrancar — imposible que
diverja de lo que ve el chat):

- Una tool por spec del registry: `search_corpus`, `recall_memory`,
  `research_topic`, `plan_task`, `run_ingestion`, `get_user_profile`,
  `get_system_status`, `list_topics`, `compile_report`, …
- Genéricas: `ipa_tool(name, args)`, `list_ipa_tools()` — funcionan aunque el
  catálogo no se haya podido obtener al arrancar.
- Tutor read-only: `tutor_focus`, `tutor_projects`, `tutor_roadmap_context`.

`research_topic` es asíncrono (igual que en el chat): devuelve "en curso" y
el material aterriza después — consultar con `list_promotions`/`get_report`.

## Variables de entorno

| Variable | Default | Efecto |
|---|---|---|
| `IPA_LLM_PROVIDER` | `ollama` | `ollama` o `exl3` (exl3 sin GPU cae a ollama) |
| `IPA_OLLAMA_MODEL` | `qwen3.5:9b-q4_K_M` | modelo del chat |
| `IPA_FORCE_CPU` | (no) | `1` desactiva GPU detection del provider; no fija globalmente EmbeddingAdapter ni reranker |
| `IPA_AUTO_RESEARCH` | `1` | auto-research ante gap de corpus (`0` desactiva) |
| `IPA_AUTO_RESEARCH_DEDUP_MINUTES` | `10` | ventana de dedup de research |
| `IPA_IDLE_LLM_LOADED_ENRICHMENT` | `1` | Tier 2 con modelo ya cargado (≥5 min idle) |
| `IPA_IDLE_DEEP_THRESHOLD_MINUTES` | `30` | umbral de idle profundo |
| `IPA_RESEARCH_REVIEW_IDLE_SECONDS` | `60` | inactividad para el review worker |
| `IPA_RETRIEVAL_TIMEOUT_SECONDS` | `60` | timeout del retrieval híbrido |
| `IPA_SEARXNG_URL` | `http://127.0.0.1:8888` | backend de búsqueda web para research (default ya inyectado por launcher/watchdog) |
| `IPA_SEARXNG_MANAGED` | `1` | `0` = el watchdog no gestiona el SearXNG local (p.ej. instancia remota) |
| `IPA_SEARXNG_CHECK_INTERVAL` | `60` | segundos entre chequeos de SearXNG del watchdog |
| `IPA_PROXY_URL` | `http://127.0.0.1:8765` | dashboard al que el MCP server proxea |
| `IPA_MCP_TIMEOUT` | `300` | timeout (s) de los HTTP calls del MCP server |
| `IPA_OLLAMA_NUM_GPU` | (auto) | capas del modelo a GPU por request (`num_gpu`); el auto-fit de Ollama es conservador — forzarlo sube el decode ~2x |
| `IPA_OLLAMA_KEEP_ALIVE` | `30m` | cuánto retiene Ollama el modelo cargado tras cada request (`-1` nunca descarga) |
| `IPA_OLLAMA_NUM_KEEP` | `2048` | tokens del INICIO que se preservan al llenarse el contexto; default de llama.cpp = 4 → el system prompt se evapora primero |
| `IPA_OLLAMA_NUM_CTX` | `6144` | contexto por request; los prompts reales miden 2.3-4.1k |
| `IPA_OLLAMA_REPEAT_LAST_N` | (servidor) | ventana del repetition penalty |
| `IPA_OLLAMA_NUM_BATCH` | (servidor) | batch de prefill por request |
| `IPA_EMBED_CACHE_SIZE` | `256` | LRU de embeddings de query (`0` desactiva) |
| `IPA_EMBED_DEVICE` | `auto` | `auto`, `cpu` o `cuda`; `cpu` fija CPU explícitamente |
| `IPA_EMBED_BATCH_CPU` / `_GPU` | `4` / `4` | batch interno del adapter; callers con batch explícito lo sobrescriben |
| `IPA_EMBED_MIN_FREE_MB` | `2048` | headroom físico mínimo para que `device=auto` seleccione CUDA |
| `IPA_EMBED_GPU_BULK` | `1` | habilita lease GPU exclusivo para backlog ≥512; `0` lo deshabilita pero no pinnea el adapter a CPU |
| `IPA_EMBED_GPU_MIN_BACKLOG` / `_WAIT_SECONDS` | `512` / `1800` | umbral y espera máxima del lease GPU bulk |
| `IPA_EMBED_PASS_CHUNKS` | `256` | chunks por pasada del drain |
| `IPA_RERANK` | `1` | stage-2 cross-encoder habilitado (`0` lo desactiva) |
| `IPA_RERANK_DEVICE` | `auto` (`cpu` en esta máquina) | `auto`, `cpu` o `cuda`; `auto` decide por VRAM física disponible. El launcher fija `cpu`: el reranker es un singleton lazy y si `auto` lo carga en CUDA con el LLM ausente, queda residente y compite por VRAM cuando el chat vuelve (EXP-008). El pin también está en la env de usuario |
| `IPA_RERANK_MIN_FREE_MB` | `2048` | headroom físico mínimo para reranker CUDA |
| `IPA_RERANK_CACHE_SIZE` | `128` | LRU de rankings del cross-encoder (`0` desactiva) |
| `IPA_RETRIEVAL_CACHE_SIZE` / `_TTL` | `64` / `300` | cache TTL de resultados de retrieval (query→hits) |
| `IPA_TOOL_CACHE_TTL` | `30` | TTL del cache de tools read-only (`0` desactiva; `get_system_status` nunca se cachea) |
| `IPA_RESPONSE_CACHE` | `0` (off) | cache de respuestas del chat: dedup exacto por (mensaje, rol, sesión) |
| `IPA_VRAM_LOCK_TTL` | `1800` | TTL del lock de VRAM ExL3↔Ollama |
| `IPA_IDLE_DEEP_ENRICHMENT` | `1` | pase Tier 2 profundo (≥30 min idle); con `IPA_T2_ENGINE=exl3` es el hogar del trabajo batch |
| `IPA_T2_ENGINE` | `exl3` | motor del pase Tier 2 profundo (`ollama` para volver al comportamiento anterior) |
| `IPA_T2_CTX` / `IPA_T2_BATCH` | `2048` / `4` | contexto y lote del pase ExL3 (batch 4 = sweet spot medido; MTP se apaga solo) |
| `IPA_T2_REVIEW_LIMIT` | `12` | docs de la cola de review por pase batched |
| `IPA_T2_ENRICH_LIMIT` / `_MIN_CHARS` | `60` / `400` | chunks por pase de `enrich_chunks` y umbral de tamaño del filtro de densidad (el corpus es uniforme ~512 chars; con 800 no selecciona nada) |
| `IPA_EXL3_FORCE_MTP` | `0` | `1` fuerza MTP aunque el batch sea > 2 (medido: thrashing) |

## Hardware: GPU o 100% CPU

- **Con GPU**: modelo estrella en CUDA, OCR y Docling acelerados.
- **Sin CUDA disponible**: chat/Tutor por Ollama (CPU), OCR/Docling en `cpu` y
  embeddings en CPU. `IPA_FORCE_CPU=1` afecta la detección de GPU del provider;
  no fija por sí solo los devices de `EmbeddingAdapter` ni del reranker.
- **Pin explícito**: `IPA_EMBED_DEVICE=cpu` fija BGE-M3 en CPU y
  `IPA_RERANK_DEVICE=cpu` fija el cross-encoder en CPU. `IPA_EMBED_GPU_BULK=0`
  o `run_embed_drain.py --cpu-only` solo deshabilitan el lease GPU bulk: con
  `device=auto`, el adapter todavía puede elegir CUDA si el gate de VRAM lo permite.

### Configuración efectiva de embeddings

`EmbeddingAdapter` usa batch 4 por default y FP32 cuando el device es CPU. El
baseline de PM-004 fijó `torch.set_num_threads(6)` en el probe; el adapter no fija
threads, aunque el default de PyTorch de este venv reporta actualmente 6. Un
caller puede sobrescribir el batch: `continuous_pipeline --lancedb-batch-size`
defaultea a 256 y lo pasa al forward; workers `lancedb_incremental.py` y Tier 2
re-embed usan 192, y el chunker semántico 16. Esos overrides no se cubren con el
microbenchmark batch 4. En cambio, `run_embed_drain.py --batch-size` controla el
flush hacia LanceDB, no el batch interno del modelo.

`BAAI/bge-m3` en el adapter actual usa FP32 en CPU / FP16 en CUDA; no hay un
switch FP8. El diseño experimental de FP8 E4M3/NVIDIA scaling para Ada está en
`knowledge/experiments/EXP-009-bge-m3-fp8-rtx4050.md` (propuesto, no ejecutado).

El reranker usa `device=auto` por default —no se asume CPU—, FP32 cuando resuelve
CPU y FP16 en CUDA. El wrapper actual hereda batch 128 de FlagReranker y pasa
`max_length=8192`; no hay tuning CPU de esos knobs validado. Ver EXP-007 para la
evidencia de calidad/latencia existente.

### Ajuste de VRAM para velocidad (tok/s)

El auto-fit de Ollama es conservador: deja ~1.4 GB de VRAM libre y manda la
mitad del modelo a CPU. Medido en RTX 4050 (6 GB) con qwen3.5:9b:

| Config | Capas GPU | decode | prefill |
|---|---|---|---|
| auto | 18/34 | 10.4 tok/s | 453 tok/s |
| `IPA_OLLAMA_NUM_GPU=28` | 28/34 | 17.3 tok/s | 653 tok/s |
| `IPA_OLLAMA_NUM_GPU=30` | 30/34 | 20.6 tok/s | 680 tok/s |
| `IPA_OLLAMA_NUM_GPU=32` | 32/34 | 9.3 tok/s | 22 tok/s (sin VRAM para compute) |

El punto de quiebre depende de la GPU — subir de a 2 y medir. Complementos:
`OLLAMA_KV_CACHE_TYPE=q8_0` (mitad de VRAM de KV), `OLLAMA_FLASH_ATTENTION=1`.
Si al matar/reiniciar Ollama quedan `llama-server.exe` huérfanos, retienen GB
de VRAM y el siguiente auto-fit manda el modelo a CPU: el watchdog los barre
solo cuando la API está caída.

### Caches y convivencia de motores

- **PT cache (Ollama)**: el prompt se arma con lo inmutable en el system y todo
  lo volátil (evidencia RAG, memoria, catálogo de tools) al final del turno
  user → el runner reusa el prefijo. Medido: 72-90% de reuse por turno
  (`prompt_eval_cached_count` en `outputs/web_dashboard/logs/llm_perf.jsonl`).
- **PT cache (ExL3)**: automático — el generator de ExLlamaV3 hashea las
  páginas de KV y reusa prefijos compartidos entre jobs. Medido: TTFT
  3.7s → 0.34s (10.9x) en la segunda generación con el mismo prompt.
- **Caches de la aplicación**: embeddings de query, rankings del reranker,
  resultados de retrieval, tools read-only y (opt-in) respuestas del chat.
  Todos con env de tamaño/TTL — ver la tabla de variables.
- **ExL3 ↔ Ollama (lock de VRAM)**: en GPUs chicas no conviven. `ExL3.load()`
  descarga los modelos de Ollama, toma `outputs/agent/vram.lock` y lo libera
  en `unload()`. Mientras ExL3 lo tiene, el chat devuelve
  "GPU ocupada por exl3" en vez de matar el batch con OOM. Un lock de un
  proceso muerto se roba solo (TTL `IPA_VRAM_LOCK_TTL`).
- **Batch ExL3 en 6 GB**: `batch_size=6` con contexto 6144 **no entra** (OOM
  al cargar). Con contexto 2048 entra pero el generator reencola jobs por
  presión de páginas (medido: 17.5 tok/s agregado con secuencias de 3 a 31
  tok/s, vs 37 tok/s single-stream). Para jobs batch usar contexto ≤2048 y
  lotes ≤3.

## Seguridad y política de datos

- Nunca commitear secretos, datos personales o corpus no autorizado.
- `.gitignore` cubre: `Landing/`, `Archive/`, `Transit/`, `models/`,
  `outputs/`, `local_archive/`, `*.db`, logs y `.env`.
- Los artefactos generados (índices, reportes) son derivados y rebuildables.
- Scraping solo con allowlists explícitas y jobs acotados.
