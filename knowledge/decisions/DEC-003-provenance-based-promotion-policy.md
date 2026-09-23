---
id: DEC-003
category: decision
status: accepted
created: 2026-09-11
updated: 2026-09-23
author: human
components: [document_store, agentic_runtime, reporter, ingestion]
tags: [provenance, promotion, configured-scrape, agent-research, user-provided, policy, decoupling, staging]
related: [PAT-001, PAT-003, EXP-003, PM-001]
supersedes: null
superseded_by: null
evidence: ["src/ipa/agentic/promotion_policy.py", "tests/test_promotion_executor.py"]
affects: ["src/ipa/agentic/promotion_executor.py", "src/ipa/agentic/promotion_policy.py", "src/ipa/reporter/**", "src/ipa/agent/research_executor.py", "scripts/operations/run_research.py", "outputs/agent/research_staging/**"]
---

# DEC-003 — Política de promoción basada en proveniencia

## Contexto

La promoción de documentos al corpus principal (`MAIN_CORPUS`) estaba acoplada al Reporter: un documento solo llegaba al corpus principal cuando un humano aprobaba un reporte completo. Esto creaba varios problemas:

1. **Cuello de botella humano**: aprobar reportes de cientos de documentos era inviable a escala.
2. **Sin distinción de origen**: los documentos de fuentes configuradas (scrape de sitios conocidos) y los de investigación del agente (búsquedas web) recibían el mismo tratamiento.
3. **Idle enrichment sin métricas reales**: el idle enrichment llamaba `curate_documents()` con interests vacíos, sin embeddings históricos y sin metadata de fuente, produciendo scores degradados (relevance=0.5, source_quality=0.5).
4. **Reporter como prerrequisito**: la promoción física requería un `report.json` aprobado, no podía ocurrir continuamente.

## Decisión

Desacoplar la promoción del Reporter mediante una política explícita basada en proveniencia:

- **`configured_scrape`**: documentos obtenidos de URLs configuradas en `configs/scrape_sites.yaml` → **auto-promoción** (sin umbral de score). La fuente es confiable por diseño.
- **`agent_research`**: documentos obtenidos por búsquedas web del agente → promoción solo si `promotion_score >= 0.70`.
- **Proveniencia desconocida**: no se promueve.

La política vive en `src/ipa/agentic/promotion_policy.py` y es auditable. La promoción física vive en `src/ipa/agentic/promotion_executor.py` y es idempotente.

El Reporter sigue existiendo para generar reportes de calidad, pero ya no es el gate de promoción. La función `promote_report_to_main()` fue eliminada; el endpoint `/api/reports/review` ahora usa `promote_documents_to_main()` directamente.

## Consecuencias

- **Gana**: promoción continua sin intervención humana para fuentes confiables; idle enrichment con métricas reales; Reporter opcional para promoción.
- **Coste**: requiere tracking de proveniencia en `DocumentStore.document_sources`; la política debe mantenerse explícita.
- **Reversibilidad**: la política es un módulo independiente; cambiar los umbrales o añadir clases de proveniencia no toca el DocumentStore ni el Reporter.

## Enmienda 2026-09-23 — el gate de novelty requiere evidencia doble

La curación alimenta esta política: un `DUPLICATE` nunca llega a la cola. El
gate "casi idéntico" se calculaba como `novelty = 1 − max_cosine` sobre **un
único vector por documento** contra el histórico de main — dominado por el
boilerplate del sitio, rechazó artículos distintos de series recurrentes
(alertas CISA semanales con CVEs diferentes, entrevistas de una serie,
announcements con template compartido). Auditoría: ~168 falsos positivos
rehabilitados y promovidos (`rehabilitate_rejected_docs.py`).

Regla vigente: el rechazo por similitud semántica exige **dos factores** —
coseno >0.95 (embedding) **y** Jaccard ≥0.85 de tokens contra el documento
histórico que produjo el máximo coseno. Un match fuzzy no verificable conserva
el documento (`REPORTER_ONLY`), nunca lo destruye. La identidad de URL/hash y
el fallback léxico (>0.95 Jaccard directo) siguen rechazando sin cambios.

- `src/ipa/reporter/reporter_curation.py` — gate de dos factores + `duplicate_of`
- `src/ipa/agentic/idle_enrichment.py` — `historical_documents` alineados
  con `historical_embeddings` para la confirmación léxica
- `tests/test_reporter.py` — 4 tests del gate (confirmación, duplicado real,
  match inverificable, fallback léxico)

## Enmienda 2026-09-23 (b) — research aterriza en staging propio + `user_provided`

La research del agente dejaba de cumplir la política: ingería **directo al
corpus principal** tras el juez por-fuente, así que el gate de score nunca se
aplicaba (la curación T1 solo veía el staging del reporter). El incidente de
proveniencia (21 docs con `source_url` ajena por el fallback `web_sources[0]`)
mostró el coste de escribir en main sin revisión.

Cambios:

- **Staging dedicado**: `outputs/agent/research_staging/` — path fijo del
  dominio agente. No se usa `active_reporter_output()/corpus`: ese puntero se
  mueve entre corridas del pipeline y el cleanup del reporter puede borrar
  runs, lo que huérfanaría docs pendientes de curación.
- `execute_research(staging_corpus_dir=...)`: ingesta, provenance,
  ingest_metadata, dirty flag y embeddings aterrizan en staging; el retrieval
  de la respuesta fusiona hits de main (hybrid) + BM25 directo sobre staging
  (el material fresco sigue respondiendo de inmediato).
- **`user_provided`**: URLs pegadas por el usuario (seeds) se registran con
  esa proveniencia → auto-promoción, igual que `configured_scrape` (fuente
  conocida por autorización directa). Respeta los gates de curación
  (DUPLICATE / INSUFFICIENT_EVIDENCE). Todo lo demás sigue `agent_research`
  con el umbral 0.70.
- T1 gana `topify_research_staging` (y T2 `deep_topify_research_staging`);
  `promotion_queue` no cambia — ya agrupaba por `source_corpus` por entry.
- Backlog residual de embeddings en staging → `run_embed_drain --corpus
  <staging>` post heavy-phase (lease propio, escala a GPU bulk si ≥512);
  sin él la promoción defería para siempre (preflight PM-004).
- La cola de review (`ingest_reviewed_doc`) usa el mismo destino vía
  `_research_ingest_corpus()` — el LLM re-review ya no es un bypass a main.
- Kill switch: `IPA_RESEARCH_STAGING=0` restaura ingesta directa a main.

## Evidencia

- `src/ipa/agentic/promotion_policy.py` — `evaluate_promotion()`, `evaluate_batch()`
- `src/ipa/agentic/promotion_executor.py` — `promote_documents_to_main()`, `process_promotion_queue()`
- `src/ipa/ingestion/provenance.py` — backfill y registro de proveniencia
- `tests/test_idle_enrichment.py` — tests de política (configured_scrape auto, agent_research >= 0.70, below threshold, unknown)
- Main corpus: 689 docs, 40,903 chunks, 40,903 vectores (sincronizados)
- Reporter corpus: 701 docs con proveniencia (666 configured_scrape, 35 agent_research)

## Por qué no es ADR

Es una decisión local y reversible: la política es un módulo que puede cambiarse sin alterar fronteras arquitectónicas. El DocumentStore sigue siendo canónico; los índices siguen siendo derivados. La separación Reporter/promoción no cambia una frontera, la hace explícita.
