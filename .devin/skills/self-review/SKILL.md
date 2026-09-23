---
name: self-review
description: Revisión crítica antes de dar por cerrado un trabajo — arquitectura, calidad, correctitud, mantenibilidad y cosecha EKS.
triggers:
  - user
  - model
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Revisá el trabajo antes de declararlo terminado. Reportá debilidades con
honestidad; no asumas que la implementación es óptima.

1. **Arquitectura y fronteras**: ¿respeta `docs/architecture/boundaries.md` y
   los records que gobiernan los paths tocados? Consultá
   `eks_governing(paths)` — incluye rejected/superseded (el cementerio).
2. **Revisá cada eje y decí cuál está flojo**, no "todo ok": naming,
   dependencias, manejo de errores, logging, tests, documentación.
3. **Verificación real, no de forma**: corré lo que corresponda —
   `.venv/Scripts/python.exe -m pytest -q` (o el subconjunto del área) y
   `scripts/validation/validate_eks.py` si tocaste EKS. Un test que solo
   chequea existencia de archivos no valida comportamiento.
4. **Tests**: ¿hay un test que fallaría sin el cambio? Si el área no tiene
   infraestructura de test, decilo — no inventes cobertura.
5. **Diff completo** (`git diff`): cambios accidentales, comentarios
   borrados, secretos, `.gitignore` o `.venv` tocados sin autorización.
6. **Cosecha de conocimiento**: ¿este trabajo generó un Experiment, Benchmark,
   Decision (DEC-*), Postmortem, Pattern o Research? Si sí, invocá el skill
   `experiment-logging` (clasifica, scaffoldea con evidencia, deja `draft` si
   falta evidencia). Si no, declaralo — no fabriques records.
7. Si estás bajo work permit, cerrá con el skill `session-closeout`.

Salida: hallazgos por severidad (alta/media/baja) con evidencia concreta
(`path:línea`, comando, salida), y explícitamente qué quedó sin verificar y
por qué.
