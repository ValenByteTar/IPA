# EKS Metadata Schema

Todo documento bajo `knowledge/` debe comenzar con este frontmatter YAML:

```yaml
---
id: DEC-001
category: decision
status: draft
created: 2026-09-05
updated: 2026-09-05
author: human
components: []
tags: []
related: []
supersedes: null
superseded_by: null
---
```

## Campos

- `id`: identificador único. Prefijos: `DEC`, `EXP`, `BM`, `PM`, `PAT`, `RES`.
- `category`: `decision`, `experiment`, `benchmark`, `postmortem`, `pattern` o `research`.
- `status`: `draft`, `proposed`, `accepted`, `rejected` o `superseded`.
- `created`, `updated`: fechas ISO `YYYY-MM-DD`; `updated` cambia en toda edición.
- `author`: humano o agente que registró el documento.
- `components`: vocabulario IPA, por ejemplo `ingestion`, `document_store`, `retrieval`, `reporter`, `tutor`, `agentic_runtime`, `mcp`, `observability`, `configuration`, `eks`.
- `tags`: términos de búsqueda.
- `related`: IDs EKS, ADRs, contratos o artefactos relacionados.
- `supersedes`, `superseded_by`: ID relacionado o `null`.

Los documentos aceptados no se editan sustancialmente. Una decisión nueva debe superseder a la anterior y enlazar ambos IDs.

## Prefijos

| Prefijo | Categoría | Carpeta |
|---|---|---|
| DEC | decision | `decisions/` |
| EXP | experiment | `experiments/` |
| BM | benchmark | `benchmarks/` |
| PM | postmortem | `postmortems/` |
| PAT | pattern | `patterns/` |
| RES | research | `research/` |
