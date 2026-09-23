---
name: experiment-logging
description: Clasifica y registra evidencia local de experimentos, benchmarks, decisiones y postmortems en EKS (scribe workflow).
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
    - Read(knowledge/**)
    - Read(outputs/**)
    - Read(docs/**)
    - Write(knowledge/**)
  deny:
    - Write(src/**)
    - Write(contracts/**)
    - Write(.venv/**)
    - Write(.gitignore)
---

Registrá conocimiento de ingeniería después de una corrida, cambio o
incidente con evidencia local. Es el workflow de escritura del EKS — el
MCP es read-only a propósito (RES-003); acá se autoría por scaffold.

1. Revisá `knowledge/_schema/metadata.md` y el template de la categoría.
2. Clasificá el resultado primario:
   - Experiment: aprendizaje puntual reproducible;
   - Benchmark: baseline congelada o gate de no-regresión;
   - Decision (`DEC-*`): decisión local o de frontera — DEC-* ES el formato
     ADR del proyecto (DEC-008); no existe `docs/adr/` para decisiones propias;
   - Postmortem: fallo con causa y prevención;
   - Pattern: solución reusable validada;
   - Research: pregunta abierta con marco de investigación;
   - Nothing: ruido, evidencia incompleta o resultado no reproducible.
3. Scaffoldeá — nunca el ID a mano:
   `.venv/Scripts/python.exe scripts/cli/eks_new.py <category> --title "..." \
     --status draft --author agent --author-model <tu modelo> \
     --evidence <paths reales> --affects <globs gobernados> \
     --trigger <origen | permit:PW-* si corrés bajo work permit>`
4. Evidencia honesta: `evidence:` debe apuntar a paths que existen en disco
   (outputs/, docs/, src/, tests/) o citar un artefacto existente en el body.
   Sin evidencia verificable el record queda `draft` — un `accepted` creado
   desde 2026-09-23 sin ella es error de validación.
5. `affects:` si el record gobierna paths concretos — sin él no entra al
   circuito `eks_governing`/permits y el conocimiento queda consultable
   pero nunca inyectado.
6. `related:` buscá links entrantes/salientes con `eks_search` antes de
   escribir — un record huérfano lo reporta `eks_report`.
7. Completá el body con secciones del template; no inventes resultados,
   métricas ni owners — si falta evidencia, `draft` o `Nothing`.
8. No edites sustancialmente documentos `accepted`; supersedelos con un
   nuevo ID (`--supersedes`) y marcá el viejo `status: superseded` +
   `superseded_by` — el validador exige el par recíproco.
9. Corré `.venv/Scripts/python.exe scripts/validation/validate_eks.py`
   antes de finalizar; si quedó un ERROR nuevo, corregilo.

La salida debe incluir path, categoría, ID, estado, evidence, affects y
relaciones creadas.
