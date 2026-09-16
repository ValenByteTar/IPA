---
id: DEC-003
category: decision
status: accepted
created: 2026-09-11
updated: 2026-09-11
author: human
components: [document_store, agentic_runtime, reporter, ingestion]
tags: [provenance, promotion, configured-scrape, agent-research, policy, decoupling]
related: [PAT-001, PAT-003, EXP-003, PM-001]
supersedes: null
superseded_by: null
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

## Evidencia

- `src/ipa/agentic/promotion_policy.py` — `evaluate_promotion()`, `evaluate_batch()`
- `src/ipa/agentic/promotion_executor.py` — `promote_documents_to_main()`, `process_promotion_queue()`
- `src/ipa/ingestion/provenance.py` — backfill y registro de proveniencia
- `tests/test_idle_enrichment.py` — tests de política (configured_scrape auto, agent_research >= 0.70, below threshold, unknown)
- Main corpus: 689 docs, 40,903 chunks, 40,903 vectores (sincronizados)
- Reporter corpus: 701 docs con proveniencia (666 configured_scrape, 35 agent_research)

## Por qué no es ADR

Es una decisión local y reversible: la política es un módulo que puede cambiarse sin alterar fronteras arquitectónicas. El DocumentStore sigue siendo canónico; los índices siguen siendo derivados. La separación Reporter/promoción no cambia una frontera, la hace explícita.
