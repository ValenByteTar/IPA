# CHANGELOG — IPA

Formato: [versión] — fecha. Estilo Keep a Changelog (resumido).

## Unreleased — 2026-09-18

### Retrieval
- **Rerank stage-2 ON por defecto** (`IPA_RERANK=0` para desactivar; acepta
  `false`/`no`/`off`). Eval E10-rerank: recall@1 0.465→0.670 (+20.5pp),
  MRR +0.141, nDCG@10 +0.117; ~+0.65s GPU / ~+0.7s CPU. Ver `EXP-007`.
- **Gate de VRAM corregido**: medía con `torch.cuda.mem_get_info()`, que en
  Windows/WDDM sobreestima la VRAM libre (reportó ~5 GB con 1.6 GB físicos
  libres) → el reranker cargaba en GPU con el LLM cargado. Ahora usa
  `nvidia-smi` (`physical_free_vram_mb()`) con fallback.
- `run_retrieval_eval.py` respeta `IPA_RERANK_DEVICE`.

### Tutor
- **Scaffold absorbente** (`_shape_units`): propuestas imperfectas del LLM ya no
  producen dead-ends — descarta concept_ids desconocidos, deduplica, completa al
  mínimo (3) y trunca a 7; el prompt dejó de invitar a reutilizar concept_ids
  (contradecía el contrato). Fallos residuales → mensaje amigable (la excepción
  va al log) + nota de transparencia si el corpus no menciona el tema.
- **Foco de roadmap cross-sesión**: click en la card (o activar un roadmap)
  apunta la sesión y persiste el foco (`tutor_focus`); sesiones nuevas/idle lo
  adoptan; rechazar lo limpia. Chip "📍 tema · unidad N/M" en el chat.
- **Tag de roadmap en resúmenes de sesión** (`[roadmap:<id> · tema · unidad N/M]`)
  → recuperable cross-sesión vía `recall_memory`.
- **Lecciones ~2x más largas**: la identidad base escopa "1-5 oraciones" al chat
  general y da excepción al rol tutor; `TUTOR_POLICY` pide explicaciones ricas;
  `max_new_tokens` de lección 768→1536. Medido: ~90→184 palabras.
- Gate del chat: aprobar un roadmap mostraba "Rechazado" (contrato
  frontend/backend: `decide_roadmap` devuelve `active`, no `approved`).
- Diagnóstico sin leak de policy ("Comenzá con un diagnóstico…" ya no se muestra
  al alumno) y sin doble punto; encabezado único en el debate.

### Agente / UI
- **Perilla "Idle T1/T2"** en el sidebar: apaga el enriquecimiento idle completo
  (gate en `_idle()`; aborta pasadas Tier 2 en vuelo). Persistida en
  `idle_enabled.json`; endpoints `GET /api/idle/status`, `POST /api/idle/toggle`.
- **Quote-reply en el chat**: seleccionar texto inserta `[respondiendo a: «…»]`
  en el input.
- MCP server: corregido el import (insertaba `src/ipa` en `sys.path` y el
  paquete local `ipa/mcp` sombreaba el SDK `mcp` — el módulo no importaba) y
  `RerankCandidate(id=…)` (campo real: `chunk_id`; el TypeError se tragaba y el
  rerank no se aplicaba). Suite MCP nueva. Docs de tools sincronizadas.

### Infra / calidad
- Deprecaciones de LanceDB resueltas (`list_tables()`, `create_index(config=FTS())`);
  warnings de la suite 78 → 18 (los restantes son de terceros).
- `tests/conftest.py`: `IPA_RERANK=0` autouse (suite hermética, sin cargar el
  cross-encoder por el default).
- 881 tests, 1 skipped (red).

## v0.1.0 — 2026-09-16

Primera versión estable consolidada: agente personal local-first, contract-first, funcional de punta a punta.

### Núcleo
- **Hybrid RAG contract-first**: `DocumentStore` canónico; Tantivy/BM25, LanceDB, embeddings, clusters, reportes y memoria como derivados rebuildables y auditables.
- Agent core compartido (DEC-002): CLI y dashboard comparten identidad, sesiones y memoria episódica (`outputs/agent/agent.db`).
- Chat con protocolo de tools acotado (máx. 3 rondas por turno), progressive tool unlocking y safety nets (anti-repetición, auto-research con dedup).

### Tutor
- State machine determinística (`tutor_runtime.py`): diagnóstico → roadmap (LLM propone, humano aprueba) → lección → assessment (JSON estructurado con abstención) → mastery persistido.
- Debate de roadmaps: feedback del alumno → re-propuesta (v+1, supersedes) con el gate humano intacto.
- Research del tutor: 15 fuentes / 300s (chat general: default 5, tope 20).
- Recuperación ante JSON malformado del LLM: retry correctivo + fallback determinístico.
- Gates re-montables en la UI (`mountPendingTutorGates`) tras cada re-render canónico.

### Adquisición
- Scraper multi-engine (requests / Playwright / auto) con OCR, patrones determinísticos por URL y ventanas por fuente.
- FastPath: parse → chunk → BM25 + LanceDB con trazabilidad (E11).
- Reporter como tool del agente (promoción desacoplada, policy por provenance).

### Idle / background
- `idle_scheduler.py`: registro de tareas con tier, prioridad, recursos (locks nombrados) y cooldowns. Tier 1 paralelo (pool de 3), Tier 2 serial preemptible.
- Consolidación automática de sesiones (resumen + hechos → cola de aprobación), con cierre de sesiones huérfanas.
- Review queue de docs rechazados (re-lectura LLM en idle, resumible).
- Enrichment L1 determinístico (topificación, curación heurística, continuidad) y Tier 2 LLM (re-etiquetado, clasificación de grises, principios).

### Infra
- Dashboard `ThreadingHTTPServer` + SSE; stores SQLite thread-bound (sin `check_same_thread=False`).
- Contrato de providers normalizado (`ipa/agent/llm_text.py`): Ollama (str) y ExL3 (`GenerationResult`).
- Provider estrella: Qwen3.5-9B EXL3 3.0bpw + MTP (extensión nativa compilada, sm_89).

### Calidad
- 854 tests, 1 skipped (red). Suite completa en verde.
- Limpieza: propuestas de test fuera de la cola de aprobaciones, scripts temporales eliminados.
