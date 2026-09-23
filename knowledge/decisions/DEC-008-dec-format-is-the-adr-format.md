---
id: DEC-008
category: decision
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: human
components: [eks, configuration]
tags: [adr, governance, decision-format, lifecycle]
related: [RES-003, DEC-002, DEC-007]
supersedes: null
superseded_by: null
affects: ["knowledge/**", "docs/DECISION_LOG.md"]
---

# DEC-008 — DEC-* es el formato ADR del proyecto

## Contexto

RES-003 declaró que "si una decisión cambia una frontera arquitectónica, vive
en `docs/adr/`". Esa carpeta nunca se materializó: las siete decisiones del
proyecto —incluidas dos de frontera explícita (DEC-002, boundaries del agent
core; DEC-007, ciclo de vida Landing→Transit→Archive)— viven como DEC-* con
frontmatter, validación y exposición vía MCP. El resultado eran dos
autoridades declaradas para lo mismo y un plano ADR vacío.

## Decisión

`DEC-*` con el frontmatter de `_schema/metadata.md` **es** el formato ADR de
IPA. Una decisión de frontera arquitectónica se registra como
`category: decision` en `knowledge/decisions/`, con el mismo ciclo de vida
que el resto del EKS (`supersedes`/`superseded_by`, nunca edición sustancial
de lo aceptado).

`docs/adr/` queda como *reference root* opcional del MCP: si en el futuro se
importan ADRs externos o legacy, se leen como documentos de referencia, no
como destino de decisiones nuevas.

## Consecuencias

- Una sola autoridad para decisiones: `knowledge/decisions/`, validada por
  `validate_eks.py` y consultable por `eks_*`.
- La plantilla `decision.md` deja de pedir "por qué no es ADR": toda DEC lo
  es. La sección pasa a documentar alcance y reversibilidad.
- Se elimina la disyuntiva "¿DEC o ADR?" al registrar una decisión de
  frontera — el criterio de registro pasa a ser único.

## Evidencia

- RES-003 (cláusula enmendada por esta decisión).
- `tools/eks_repository.py` — `reference_roots` sigue soportando `docs/adr`
  para ADRs externos; ningún documento local lo usa.
- `docs/DECISION_LOG.md` — log pre-EKS congelado el 2026-09-06; sus
  decisiones vigentes ya están reflejadas en records EKS.

## Alcance

Frontera tocada: gobernanza del conocimiento de ingeniería (dónde vive la
autoridad de una decisión). Reversible: si el proyecto adopta tooling ADR
externo, una DEC futura supersede a esta y `docs/adr/` pasa de reference
root a destino.
