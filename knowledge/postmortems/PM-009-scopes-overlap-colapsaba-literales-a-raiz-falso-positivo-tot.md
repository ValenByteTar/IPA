---
id: PM-009
category: postmortem
status: accepted
created: 2026-10-02
updated: 2026-10-02
author: agent
components: [eks]
tags: [permits, tooling, scopes, glob, port-riapp]
related: [PM-008, PAT-009]
supersedes: null
superseded_by: null
affects: [tools/eks_repository.py, tests/test_eks.py]
evidence:
  - tools/eks_repository.py
  - tests/test_eks.py
author_model: SWE-2 (Devin, sesion pumped-pine)
trigger: permit:PW-20261002-02
---

# PM-009 — scopes_overlap colapsaba literales a raíz: falso positivo total de conflictos (port RIAPP)

## Impacto

En RIAPP (PM-001/PM-003/PM-004 allá), un permit `exclusive` cuyo scope
incluía un archivo literal de raíz (`AGENTS.md`, `pyproject.toml`)
bloqueaba **todos los acquires de todas las demás sesiones** — tres agentes
quedaron frenados con "scope overlaps an active exclusive permit" pese a
tener scopes disjuntos por diseño, y resolvieron con `--force`.

El bug existía idéntico en la copia de IPA (`tools/eks_repository.py`) —
nunca explotó acá porque los permits de IPA casi no usaban literales de
raíz.

## Causa raíz

`scope_prefix("AGENTS.md")` devuelve `""` (literal sin wildcard y sin `/` —
raíz implícita). En `scopes_overlap`, `a.startswith(b) or b.startswith(a)`
con prefijo `""` es True contra cualquier scope → el literal de archivo se
comportaba como `**`. Falso positivo total del chequeo de solape.

## Corrección

`_is_literal()` + match exacto para literales en `scopes_overlap`
(`tools/eks_repository.py`):

- literal vs literal: igualdad o contención directorio/archivo (tras
  `rstrip("/")`).
- literal vs glob: `glob_match` real, no prefix.
- glob vs glob: el chequeo de prefijos original (conservador, over-report
  es la dirección segura).

## Prevención

`test_scopes_overlap_literals_match_exactly_not_as_root_prefix` y
`test_permit_literals_do_not_conflict_with_disjoint_scopes` (end-to-end en
el store) en `tests/test_eks.py`.

## Lección reutilizable

Los operadores sobre "prefijo literal de un glob" deben definir qué
significa el string vacío — `""` como prefijo no es "raíz del repo" sino
"sin directorio", y cualquier `startswith` lo convierte en comodín
universal. Los scopes a nivel de archivo (recomendados por PM-008) hacen
este caso común, no exótico.
