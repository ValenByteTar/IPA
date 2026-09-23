---
name: adr-proposal
description: Prepara propuestas de decisión arquitectónica (DEC-*) cuando un cambio toca una frontera de IPA.
triggers:
  - user
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Read(knowledge/**)
    - Read(docs/**)
    - Read(contracts/**)
---

Prepará una propuesta de decisión sin aceptarla automáticamente.

1. DEC-* ES el formato ADR del proyecto (DEC-008): las decisiones de
   frontera viven en `knowledge/decisions/` con frontmatter EKS, no en
   `docs/adr/` (ese path solo es reference root del MCP para ADRs
   externos o legacy importados).
2. Leé `knowledge/_schema/metadata.md`, `docs/` y las DEC existentes.
3. Buscá colisiones, supersession y decisiones relacionadas
   (`eks_search` / `eks_governing` sobre los paths que toca el cambio).
4. Verificá que exista evidencia local: test, benchmark, experiment o
   postmortem — sin evidencia verificable el record queda `proposed`.
5. Redactá el borrador con `scripts/cli/eks_new.py decision --status proposed`
   incluyendo `--evidence`, `--affects` (paths que gobierna) y
   `--supersedes` si reemplaza una DEC vigente.
6. No edites ni aceptes una DEC existente; no escribas el borrador como
   `accepted` — la promoción requiere aprobación humana.

La respuesta debe terminar con:

```text
Estado: Propuesto — requiere aprobación humana.
```
