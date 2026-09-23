---
id: DEC-004
category: decision
status: accepted
created: 2026-09-11
updated: 2026-09-23
author: human
components: [agent_core, reporter, dashboard]
tags: [reporter, agent-tool, compile_report, decoupling, pipeline]
related: [DEC-003, EXP-003, EXP-006, PAT-001, PAT-003]
supersedes: null
superseded_by: null
affects: ["src/ipa/reporter/**", "src/ipa/agent/compile_report_executor.py"]
---

# DEC-004 — Reporter como tool del agente (compile_report)

## Contexto

El Reporter era un pipeline batch que corría automáticamente desde el dashboard (`run_full_pipeline`) o desde CLI (`run_reporter.py`). Esto creaba un acoplamiento innecesario: el pipeline de ingesta dependía del Reporter para generar reportes, y el Reporter era la única forma de producir análisis fino.

## Decisión

Replantear el Reporter como una **tool que el agente invoca**, no como un pipeline automático:

1. **Nueva tool `compile_report`** (`src/ipa/agent/compile_report_executor.py`): el agente selecciona documentos (via `search_corpus` + `research_topic`) y llama `compile_report` con un set de `document_ids`. La tool corre curation + topic discovery + report building sobre esos documentos específicos, escribe a un output aislado (`outputs/reporter/agent/`), y retorna un `ToolResult` estructurado con report metadata + citations.

2. **Pipeline sin Reporter**: `run_full_pipeline()` en `server.py` ahora hace solo scraper + FastPath (indexing BM25 + LanceDB). No lanza el Reporter. El parámetro `skip_reporter` se removió.

3. **Endpoint `/api/reporter/run` deshabilitado**: retorna 410 Gone. El Reporter no se puede lanzar desde el dashboard sin agente.

4. **`scripts/cli/run_reporter.py` deprecado**: imprime warning y exits. Se mantiene para compatibilidad pero será removido.

## Invariantes preservados

- `DocumentStore` sigue siendo canónico (PAT-001). La tool solo lee.
- No hay ingesta ni promoción dentro de la tool (DEC-003).
- No muta el main corpus.
- Proveniencia se preserva desde `document_sources`.
- La tool es determinística (sin LLM). Futuro: labels LLM opcionales.
- Output es un artefacto aislado bajo `outputs/reporter/agent/`.
- Contratos `ToolCall`/`ToolResult` se emiten para auditoría.

## Tool contract

```
Input:
  document_ids: list[str] (required, bounded 200)
  topic: str (optional)
  period_start/period_end/period_label: str (optional, ISO)
  interests: list[str] (optional)
  similarity_threshold: float (default 0.52)
  min_documents: int (default 2)
  allow_singletons: bool (default True)
  output_dir: str (optional)

Output (ToolResult):
  report_id, report_path, markdown_path, output_dir
  document_count, selected_count, skipped_count
  category_count, parent_category_count
  curation_summary, categories, uncertainties
  source_refs (one per selected document)
```

## Tests

8 tests en `tests/test_compile_report.py`:
- Tool registration in TOOL_NAMES
- Basic report compilation from document IDs
- Empty document_ids raises ValueError
- Nonexistent docs skipped gracefully
- No corpus raises ValueError
- Does not mutate main corpus
- Interests-based curation
- Produces report.json + report.md + reporter.db

## Consecuencias

- El agente ahora controla cuándo y sobre qué documentos produce reportes.
- El pipeline de ingesta es más simple y rápido (sin Reporter batch).
- El Reporter deja de ser un gate o dependencia del pipeline.
- Futuro: LLM labels opcionales para topics, deep_dive integration.
