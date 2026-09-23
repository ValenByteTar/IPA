---
name: rag-component-development
description: Diseñar e implementar componentes de la capa RAG sin violar las fronteras (retrieval, ranking, contexto, generación, orquestación) ni duplicar índices.
triggers:
  - user
  - model
allowed-tools:
  - read
  - grep
  - glob
  - exec
  - edit
  - write
permissions:
  allow:
    - Read(**)
  deny:
    - Write(contracts/**)
    - Write(.gitignore)
    - Write(.venv/**)
---

Antes de diseñar un componente nuevo:

1. **Construí contexto**: skill `eks-engineering-brief` (o `eks_context` vía
   MCP) y leé `docs/architecture/boundaries.md` + `docs/architecture/retrieval.md`.
   Reusá lo que ya existe antes de agregar una pieza.
2. **Nunca mezcles responsabilidades.** El pipeline está partido y debe
   seguir estándolo: interpretación de query (planner) → retrieval →
   ranking/evidencia → context builder → generación → verificación →
   próxima acción. Cada una es un componente independiente.
3. **Fronteras concretas que ya existen y no se rompen**:
   - `DocumentStore` es la fuente de verdad; los índices (FTS5/Tantivy,
     LanceDB/sqlite-vec) son **vistas derivadas** y reconstruibles (PAT-001).
   - `chunks.text` es canónico; el enriquecimiento vive en
     `metadata.enrichment.enriched_text` (PAT-008).
   - Los parsers no importan embedding ni enrichment (PAT-003).
   - `contracts/` es autoridad: un componente nuevo declara su contrato y su
     registro en `contract_vocabulary.json` antes de tener tabla o store.
   - No crees un índice derivado sin consumidor (DEC-002).
4. **Nada de índices duplicados ni de reimplementar retrieval dentro de una
   superficie** (PM-001/PM-003): el MCP y el dashboard son clientes/proxies,
   no segundos runtimes de dominio.
5. **Diseñá para reuso** cuando el sistema evolucione hacia Agentic RAG:
   QueryIR → EvidenceSet → ContextPackage (PAT-004).
6. Si el componente toca una frontera, proponé una DEC (`adr-proposal`); si
   introduce un patrón reusable, un `PAT-*`.

Salida: componente propuesto, responsabilidad única, contrato afectado,
fronteras respetadas y evidencia (tests) de que no se mezclaron capas.
