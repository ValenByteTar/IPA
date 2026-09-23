---
id: RES-003
category: research
status: accepted
created: 2026-09-05
updated: 2026-09-23
author: human
components: [eks, agentic_runtime, knowledge_runtime, mcp, configuration]
tags: [engineering-memory, dev-time, runtime, separation, provenance]
related: [PAT-001, RES-002, DEC-008]
supersedes: null
superseded_by: null
evidence: ["tools/eks_repository.py", "tests/test_eks.py"]
affects: ["knowledge/**", "tools/eks_*"]
---

# RES-003 — Separación EKS, Knowledge runtime y Agent Runtime

## Tema

Definir cómo conviven la memoria de ingeniería, el conocimiento del corpus y el runtime que lo consume.

## Modelo

```text
EKS dev-time
  decisiones, experimentos, benchmarks, patterns, research
             ↓
      Devin / Cascade

IPA Knowledge runtime
  documentos, chunks, índices, enrichment, provenance
             ↓
      Reporter / Agent Runtime / Tutor
```

## Conclusiones

- EKS no debe ser una segunda base vectorial del corpus.
- El Knowledge runtime no debe guardar decisiones del proceso de desarrollo.
- El Agent Runtime puede reutilizar contratos y adapters de IPA, pero no debe leer EKS para resolver consultas de usuario.
- El MCP EKS debe ser un servidor separado, local y read-only.
- ~~ADRs, si aparecen, siguen en `docs/adr/`~~ **Enmendado 2026-09-23**: DEC-* con frontmatter ES el formato ADR del proyecto (DEC-008). `docs/adr/` queda como reference root opcional del MCP para ADRs externos; las decisiones de frontera se registran como `category: decision` en `knowledge/decisions/`.

## Estado

Aceptado como boundary de implementación EKS V1. Los detalles de memoria episódica, identidad y navegación permanecen abiertos y requieren experimentos separados.
