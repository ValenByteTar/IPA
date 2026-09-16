---
id: PAT-005
category: pattern
status: accepted
created: 2026-09-11
updated: 2026-09-11
author: human
components: [document_store, ingestion, agentic_runtime, provenance]
tags: [provenance, document_sources, configured_scrape, agent_research, canonical, backfill]
related: [PAT-001, PAT-003, DEC-003]
supersedes: null
superseded_by: null
---

# PAT-005 — Proveniencia canónica en DocumentStore

## Problema

El `DocumentStore` almacenaba documentos y chunks pero no de dónde vinieron. La metadata de fuente (URL, dominio, quality_score, clase de proveniencia) vivía en el `ReporterStore.document_metadata`, acoplada al Reporter. Esto significaba:

1. Sin Reporter, no había forma de saber si un documento venía de un scrape configurado o de una búsqueda del agente.
2. El idle enrichment no podía aplicar una política de promoción basada en proveniencia.
3. La proveniencia se perdía cuando un documento se promovía al corpus principal.

## Solución

Extender el `DocumentStore` con una tabla `document_sources` que registra la proveniencia de cada documento:

```sql
CREATE TABLE IF NOT EXISTS document_sources (
    document_id   TEXT PRIMARY KEY,
    source_url    TEXT,
    source_domain TEXT,
    provenance    TEXT NOT NULL,  -- configured_scrape | agent_research
    quality_score REAL,
    recorded_at   TEXT NOT NULL
);
```

La proveniencia se registra en el momento de la ingesta (o se backfilla después):

- **`configured_scrape`**: derivada de `scrape_report.json` + `configs/scrape_sites.yaml`. El backfill matchea por `source_uri` del LandingZone → `artifact_id` → `document_id`.
- **`agent_research`**: registrada por `research_executor.py` después de la ingesta, matcheando por `source_uri` → `web_source.source_url`.

La tabla es un índice derivado: puede reconstruirse desde `scrape_report.json` + `reporter.db` + `landing.db`. El `DocumentStore` sigue siendo la autoridad canónica para documentos y chunks; `document_sources` es metadata adjunta.

## Trade-offs

- **Gana**: la proveniencia sobrevive la promoción al corpus principal; la política de promoción no necesita al Reporter; el idle enrichment tiene metadata real.
- **Coste**: un paso extra de backfill para corpus existentes; el matching por path puede fallar si los archivos se mueven.
- **Limitación**: el `content_hash` del scraper hashea el texto extraído, no el archivo binario, por lo que el matching debe ser por path del LandingZone, no por hash.

## Ejemplos

- `src/ipa/storage/document_store.py` — tabla `document_sources`, métodos `put_source()`, `get_source()`, `all_sources()`, `sources_by_provenance()`
- `src/ipa/ingestion/provenance.py` — `backfill_from_scrape_report()`, `backfill_from_reporter_store()`, `record_agent_research()`
- `src/ipa/agent/research_executor.py` — registro de `agent_research` después de ingesta
- `tests/test_idle_enrichment.py` — tests de `document_sources` (put/get, all, by_provenance)
