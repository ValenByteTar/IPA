# Engineering Knowledge System (EKS)

EKS es la memoria de ingeniería de IPA en tiempo de desarrollo. Conserva decisiones, experimentos, benchmarks, postmortems, patrones y research para que humanos y agentes reutilicen contexto sin reconstruir la historia del proyecto.

EKS no es:

- el Knowledge System runtime;
- el corpus ingerido por IPA;
- una fuente de verdad para `contracts/`;
- el MCP de búsqueda de documentos;
- una segunda carpeta de ADRs.

## Estructura

```text
knowledge/
  _schema/metadata.md
  _templates/
  decisions/
  experiments/
  benchmarks/
  postmortems/
  patterns/
  research/
```

DEC-* es el formato ADR del proyecto (DEC-008): una decisión de frontera se
registra como `category: decision` en `decisions/` — no hay `docs/adr/` para
decisiones propias. El MCP puede aceptar roots externos de solo lectura vía
`IPA_EKS_REFERENCE_ROOTS` (sin default: `docs/adr/` no existe en este repo,
así que no se registra un root muerto).

## Ciclo de uso

1. Antes de un cambio no trivial: construir contexto de ingeniería
   (`eks_context` vía MCP, o el skill `eks-engineering-brief`).
2. Después de un experimento o benchmark: registrar la evidencia o declarar
   que no genera conocimiento persistente. Scaffold:
   `scripts/cli/eks_new.py <category> --title "..."` (asigna id, renderiza
   frontmatter desde `_templates/`, valida).
3. Una decisión nueva que reemplaza otra se registra como DEC-* y enlaza
   `supersedes`/`superseded_by` en ambos lados (el validador lo exige).
4. Consultar EKS mediante el servidor MCP dev-time, que es separado del MCP
   del corpus y sólo lectura (`eks_list`, `eks_get`, `eks_search`,
   `eks_context`, `eks_governing`, `eks_report`). `eks_governing(paths)`
   devuelve los records cuyos globs `affects` cubren esos archivos —
   incluidos rejected/superseded (el cementerio evita reintentos).
5. Sesiones paralelas: work permits dev-time (PAT-009) — ver
   `scripts/cli/permit.py`. Un permiso se emite con los records que
   gobiernan su scope; los records creados bajo permiso declaran
   `trigger: permit:PW-*`.
6. Hygiene periódica: `scripts/validation/validate_eks.py` (forma) y
   `scripts/operations/eks_report.py` (contenido: propuestas envejecidas,
   records sin referencias entrantes ni evidencia, cobertura por
   componente, zonas calientes, artefactos citados que ya no existen).

La V1 comienza sin documentos históricos importados desde otros proyectos. Los documentos futuros deben basarse en evidencia local reproducible.
