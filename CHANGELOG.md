# CHANGELOG — IPA

Formato: [versión] — fecha. Estilo Keep a Changelog (resumido).

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
