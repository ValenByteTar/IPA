---
name: engineering-context-builder
description: Construye un brief de ingeniería consultando ADRs, EKS y evidencia local antes de cambios no triviales.
triggers:
  - user
  - model
allowed-tools:
  - read
  - grep
  - glob
---

Construí un Engineering Brief antes de implementar un cambio no trivial.

1. Extraé componentes, riesgos, términos y si hay una frontera arquitectónica involucrada.
2. Consultá el MCP EKS si está disponible (`eks_context` o `eks_search`). Si no está disponible, leé `knowledge/` y `docs/adr/` directamente.
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
