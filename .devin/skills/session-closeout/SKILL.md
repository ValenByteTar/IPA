---
name: session-closeout
description: Protocolo de cierre de jornada/sesión — cierra work permits y cosecha el conocimiento en EKS drafts.
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
    - Write(knowledge/**)
  deny:
    - Write(src/**)
    - Write(contracts/**)
---

Protocolo de cierre. El closeout del permiso es donde el EKS captura el
conocimiento de la sesión — sin este paso el aprendizaje se evapora.

1. Inventariá la sesión:
   `.venv/Scripts/python.exe scripts/cli/permit.py list`
   Identificá los permisos de esta sesión (field `session`).
2. Por cada permiso activo, reconstruí qué se tocó: `git status`/`git diff`
   limitado a su scope, más lo que el scope declaraba.
3. Cosechá el conocimiento — para cada hallazgo durable, invocá el skill
   `experiment-logging` (clasifica, scaffoldea con evidencia y `affects`,
   deja `draft` si falta evidencia). Si no hubo conocimiento durable,
   declaralo — no fabriques records.
4. Cerrá cada permiso con su cosecha:
   `.venv/Scripts/python.exe scripts/cli/permit.py close --permit PW-... \
     --notes "<qué se hizo>" --eks-draft <ID del draft principal>`
5. Hygiene final:
   `.venv/Scripts/python.exe scripts/validation/validate_eks.py`
   `.venv/Scripts/python.exe scripts/operations/eks_report.py`
   Reportá drafts sin evidencia, huérfanos nuevos y warnings introducidos
   por esta sesión.
6. Resumen final: permisos cerrados, drafts creados (IDs), deuda detectada
   (drafts sin evidencia >7d, records afectados que convendría revisar).

Nunca cierres un permiso de OTRA sesión — solo `close` de los tuyos; los
ajenos se reportan al usuario.
