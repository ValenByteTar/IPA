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

Los ADRs, si una decisión cambia una frontera arquitectónica, viven únicamente en `docs/adr/`. EKS puede relacionarse con ellos mediante `related`.

## Ciclo de uso

1. Antes de un cambio no trivial: construir contexto de ingeniería.
2. Después de un experimento o benchmark: registrar la evidencia o declarar que no genera conocimiento persistente.
3. Si aparece una frontera arquitectónica: proponer un ADR, sin aceptarlo automáticamente.
4. Consultar EKS mediante el servidor MCP dev-time, que es separado del MCP del corpus y sólo lectura.

La V1 comienza sin documentos históricos importados desde otros proyectos. Los documentos futuros deben basarse en evidencia local reproducible.
