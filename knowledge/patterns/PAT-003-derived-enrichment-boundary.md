---
id: PAT-003
category: pattern
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: human
components: [enrichment, provenance, retrieval, document_store]
tags: [canonical-text, derived-data, input-hash, fingerprint]
related: [EXP-001, PAT-001]
supersedes: null
superseded_by: null
---

# PAT-003 — Boundary de enrichment derivado

## Problema

Un summary, synthetic query o claim puede confundirse con texto de fuente si se persiste en el mismo campo o se pierde su origen.

## Solución

Conservar separadas las capas:

```text
canonical_text
retrieval_enrichment
pedagogical_enrichment
knowledge_claims
```

Cada derivado debe registrar input hash, generador/model fingerprint y estado de validación. El enrichment puede mejorar retrieval, pero no reemplaza ni modifica el texto canónico.

## Trade-offs

Aumenta metadata y lifecycle, pero permite rollback, auditoría y comparación de estrategias.

## Ejemplos

- `src/res023_lab/enrichment.py`
- `src/res023_lab/reporter_contracts.py`
- `knowledge/experiments/EXP-001-selective-enrichment.md`.
