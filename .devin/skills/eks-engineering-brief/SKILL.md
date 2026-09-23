---
name: eks-engineering-brief
description: Construye un brief de ingeniería consultando EKS (ADRs/DEC, benchmarks, evidencia local) antes de cambios no triviales.
triggers:
  - user
  - model
allowed-tools:
  - read
  - grep
  - glob
---

Construí un Engineering Brief antes de implementar un cambio no trivial.

Nota de nombre: esta skill se llama `eks-engineering-brief` a propósito —
existe otra `engineering-context-builder` en la config global de Windsurf
(`~/.codeium/windsurf/skills/`) con otro contrato de salida, y el nombre
compartido hacía que esa ganara. No renombres esta de vuelta.

0. Scope primero: identificá los paths que el cambio va a tocar y consultá
   `eks_governing(paths)` — incluye rejected/superseded (el cementerio
   evita reintentos). Si hay permisos de trabajo activos, verificá que el
   scope no solape: `permit.py check --scope <globs>`.
1. Extraé componentes, riesgos, términos y si hay una frontera arquitectónica involucrada.
2. Consultá el MCP EKS si está disponible (`eks_context` o `eks_search`). Si no está disponible, leé `knowledge/` directamente (DEC-* es el formato ADR del proyecto, DEC-008).
3. Priorizá conocimiento local aceptado y benchmarks vigentes; después decisiones, patterns, experiments y research abiertos.
4. No copies documentos completos sin filtrar relevancia.
5. Si EKS está vacío o no hay conocimiento aplicable, declaralo explícitamente.
6. No edites archivos ni implementes cambios.

Devolvé exactamente estas secciones:

```markdown
## Engineering context
### ADRs aplicables
### Benchmarks / baselines
### Experiments previos
### Decisions / Patterns
### Research
### Constraints
### Riesgos
### Huecos de conocimiento
### Estrategia recomendada
```

No inventes IDs, métricas, owners ni decisiones. Diferenciá evidencia local de referencias externas.
