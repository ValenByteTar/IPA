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
# Con orquestador de consola al lado
.\start_ipa_dashboard.bat -StartOrchestrator

# Manual
.venv\Scripts\python.exe scripts\operations\web_dashboard.py --host 127.0.0.1 --port 8765
.venv\Scripts\python.exe scripts\operations\orchestrator.py --no-scraper --no-hammer

# CLI del agente (misma identidad y memoria que el dashboard)
.venv\Scripts\python.exe scripts\cli\agent.py chat
.venv\Scripts\python.exe scripts\cli\agent.py chat --role tutor --llm -m "mensaje"
```

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
  principios abstractos. Solo con idle profundo (≥30 min puede cargar el
  modelo) o aprovechando un modelo ya cargado (≥5 min quieto). Preemptible
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
| `IPA_FORCE_CPU` | (no) | `1` fuerza modo 100% CPU |
| `IPA_AUTO_RESEARCH` | `1` | auto-research ante gap de corpus (`0` desactiva) |
| `IPA_AUTO_RESEARCH_DEDUP_MINUTES` | `10` | ventana de dedup de research |
| `IPA_IDLE_DEEP_ENRICHMENT` | `0` | Tier 2 puede cargar el modelo (idle ≥30 min) |
| `IPA_IDLE_LLM_LOADED_ENRICHMENT` | `1` | Tier 2 con modelo ya cargado (≥5 min idle) |
| `IPA_IDLE_DEEP_THRESHOLD_MINUTES` | `30` | umbral de idle profundo |
| `IPA_RESEARCH_REVIEW_IDLE_SECONDS` | `60` | inactividad para el review worker |
| `IPA_RETRIEVAL_TIMEOUT_SECONDS` | `60` | timeout del retrieval híbrido |
| `IPA_SEARXNG_URL` | `http://127.0.0.1:8888` | backend de búsqueda web para research (default ya inyectado por launcher/watchdog) |
| `IPA_SEARXNG_MANAGED` | `1` | `0` = el watchdog no gestiona el SearXNG local (p.ej. instancia remota) |
| `IPA_SEARXNG_CHECK_INTERVAL` | `60` | segundos entre chequeos de SearXNG del watchdog |
| `IPA_PROXY_URL` | `http://127.0.0.1:8765` | dashboard al que el MCP server proxea |
| `IPA_MCP_TIMEOUT` | `300` | timeout (s) de los HTTP calls del MCP server |

## Hardware: GPU o 100% CPU

- **Con GPU**: modelo estrella en CUDA, OCR y Docling acelerados.
- **Sin GPU**: fallback automático — chat/Tutor por Ollama (CPU), OCR y
  Docling en `cpu`, embeddings en CPU. Nada falla al boot; `IPA_FORCE_CPU=1`
  lo fuerza explícitamente.

## Seguridad y política de datos

- Nunca commitear secretos, datos personales o corpus no autorizado.
- `.gitignore` cubre: `Landing/`, `Archive/`, `Transit/`, `models/`,
  `outputs/`, `local_archive/`, `*.db`, logs y `.env`.
- Los artefactos generados (índices, reportes) son derivados y rebuildables.
- Scraping solo con allowlists explícitas y jobs acotados.
