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
# opcionales (governance 2026-09-23):
affects: []        # globs repo-relativos que este record gobierna
evidence: []       # paths que prueban el record (outputs/, docs/, src/)
author_model: null # modelo/agente cuando author=agent
trigger: null      # qué originó el record (ej. "permit:PW-20260923-01")
---
```

## Campos

- `id`: identificador único. Prefijos: `DEC`, `EXP`, `BM`, `PM`, `PAT`, `RES`.
- `category`: `decision`, `experiment`, `benchmark`, `postmortem`, `pattern` o `research`.
- `status`: `draft`, `proposed`, `accepted`, `rejected` o `superseded`.
- `created`, `updated`: fechas ISO `YYYY-MM-DD`; `updated` cambia en toda edición.
- `author`: humano o agente que registró el documento.
- `author_model` (opcional): qué modelo/agente cuando `author: agent` —
  auditoría post-hoc del decisor. Records con `author: agent` creados desde
  el 2026-09-23 sin `author_model` generan warning.
- `trigger` (opcional): qué originó el record. Convención: `permit:PW-*`
  cuando el record sale de una sesión bajo work permit (`permit.py`).
- `components`: vocabulario controlado en `_schema/components.json`. El
  validador advierte sobre nombres desconocidos y sobre aliases (usar el
  nombre canónico para que el filtrado por componente no se degrade).
  `groups` agrupa componentes específicos bajo un bucket genérico
  (`indexes` → `lexical_index`/`vector_index`; `ingestion` → `fast_path`/
  `parsing`/`chunking`/`landing_zone`/`acquisition`; `memory` →
  `strategic_memory`/`user_model`/`skill_library`/`uncertainty`): filtrar
  por el bucket alcanza a los records etiquetados con cualquier miembro y
  viceversa (`EKSRepository.component_matches`). Etiquetar con el
  componente específico cuando exista; el genérico queda para lo
  transversal.
- `tags`: términos de búsqueda.
- `related`: IDs EKS, ADRs, contratos o artefactos relacionados.
- `affects` (opcional): globs estilo gitignore relativos al repo que este
  record gobierna (`src/ipa/agentic/**`). Alimenta `eks_governing` /
  `permit.py acquire`: editar un path afectado activa el record, incluidos
  los `rejected`/`superseded` (el cementerio evita reintentos). Un glob que
  no matchea ningún archivo genera warning — salvo globs bajo `outputs/`,
  que son paths de runtime transitorios.
- `evidence` (opcional): paths que prueban el record. Regla de promoción
  graduada: un record `accepted` creado desde el 2026-09-23 **requiere**
  `evidence` no vacía (o cita `outputs/`/`docs/` existente en el body) —
  error de validación. Records anteriores: warning. Paths de `evidence`
  inexistentes en disco: error para records nuevos, warning para legados.
- `supersedes`, `superseded_by`: ID relacionado o `null`.

Los documentos aceptados no se editan sustancialmente (un addendum fechado
sí es admisible). Una decisión nueva debe superseder a la anterior y enlazar
ambos IDs: el nuevo declara `supersedes` y el viejo pasa a
`status: superseded` + `superseded_by` — el validador exige el par
recíproco.

DEC-* es el formato ADR del proyecto (DEC-008): una decisión de frontera se
registra como `category: decision` con este frontmatter, no en un documento
aparte.

## Prefijos

| Prefijo | Categoría | Carpeta |
|---|---|---|
| DEC | decision | `decisions/` |
| EXP | experiment | `experiments/` |
| BM | benchmark | `benchmarks/` |
| PM | postmortem | `postmortems/` |
| PAT | pattern | `patterns/` |
| RES | research | `research/` |
