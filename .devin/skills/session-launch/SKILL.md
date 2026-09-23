---
name: session-launch
description: Ventanilla de emisión — antes de lanzar sesiones paralelas, revisa permisos activos, hot zones y propone scopes disjuntos.
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
    - Read(outputs/**)
---

Protocolo previo a lanzar trabajo en paralelo (3-4 sesiones Devin sobre el
mismo working tree). El objetivo: scopes disjuntos y sin solapes con
permisos vivos — la colisión se evita en la ventanilla, no en el edit.

1. Leé el estado actual:
   `.venv/Scripts/python.exe scripts/cli/permit.py list`  (activos y scopes)
   `.venv/Scripts/python.exe scripts/operations/eks_report.py`
   (sección "Hot zones" — scopes con ≥4 records gobernantes son zonas de
   alta gobernanza: mayor superficie de conflicto).
2. Por cada tarea planeada, estimá el scope de globs (dirs/archivos que va
   a tocar) y verificá:
   `.venv/Scripts/python.exe scripts/cli/permit.py check --scope <globs>`
   Conflicto con permiso `exclusive` vivo → la tarea espera o se re-scopea.
3. Proponé la asignación: tabla tarea → scope → tipo de permiso
   (`exclusive` para escritura, `survey` para investigación read-only).
   Dos tareas que convergen en un hot zone → mismo scope = misma sesión,
   o worktrees en vez de permisos.
4. Entregá para cada sesión el comando de arranque exacto:
   `.venv/Scripts/python.exe scripts/cli/permit.py acquire --session <id> \
     --scope "<globs>" --task "<desc>" --type <tipo>`
   y recordá: el acquire adjunta los records EKS que gobiernan ese scope —
   la sesión arranca ya conociendo las decisiones que le aplican.
5. Si una tarea no puede acotarse a globs concretos (exploración abierta),
   recomendá `advisory` en vez de `exclusive` — declara intención sin
   bloquear.

Salida: tabla de asignación + comandos acquire por sesión + conflictos
detectados que requieren decisión del usuario.
