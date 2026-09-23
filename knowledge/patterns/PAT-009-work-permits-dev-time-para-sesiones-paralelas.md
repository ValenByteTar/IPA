---
id: PAT-009
category: pattern
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [eks, operations, agentic_runtime]
tags: [work-permit, ptw, lease, parallel-sessions, coordination]
related: [PAT-007, DEC-010, RES-003]
supersedes: null
superseded_by: null
affects: [outputs/devin/permits/**, tools/work_permits.py, scripts/cli/permit.py, .devin/hooks.v1.json, scripts/hooks/permit_guard.py]
evidence: [tools/work_permits.py, scripts/hooks/permit_guard.py]
author_model: swe-2
trigger: session-review
---

# PAT-009 — Work permits dev-time para sesiones paralelas

## Problema

El usuario corre jornadas cortas e intensas con 3-4 sesiones Devin en
paralelo sobre el mismo working tree. Riesgo real: dos agentes editan los
mismos archivos y se pisan en silencio — el conflicto solo aparece como
diff roto al final, sin memoria de qué scope tenía cada quién.

## Solución

Permit-to-Work (PTW), patrón industrial portado a dev-time. Misma
disciplina de leases que PAT-007 (vram.lock, tier0.lock) pero para
sesiones: un permiso autoriza un scope de globs por un TTL, se emite con
las decisiones EKS que gobiernan ese scope adjuntas como "precautions", y
se cierra con notas + puntero a draft EKS — el closeout es el punto de
captura de conocimiento de la sesión.

- `tools/work_permits.py` + `scripts/cli/permit.py`: `acquire` (rechaza
  si un exclusive activo solapa el scope; `survey`/`advisory` no bloquean),
  `check`, `list`, `heartbeat`, `close --notes --eks-draft`, `close-session`.
- Estado en `outputs/devin/permits/*.json` — gitignored: la coordinación
  es estado efímero, no entra al EKS (RES-003).
- Enforcement real vía `.devin/hooks.v1.json` → `scripts/hooks/permit_guard.py`:
  PreToolUse sobre edit/write bloquea paths bajo permiso exclusivo ajeno e
  inyecta los records que gobiernan el path (una vez por record por sesión);
  SessionStart lista permisos activos; SessionEnd cierra los de la sesión;
  Stop recuerda cerrar permisos abiertos; PostCompaction los re-inyecta.
- Disciplina declarativa: `.devin/rules/eks-workflow.md` (siempre activa)
  + skills `session-closeout` (cierre con cosecha EKS) y `session-launch`
  (ventanilla previa al paralelismo: permisos activos + hot zones +
  scopes disjuntos).
- `affects` en el frontmatter EKS es lo que alimenta las precautions:
  un permiso sobre `src/ipa/agentic/**` llega con DEC-010/PM-004/PAT-007/008
  adjuntos. ≥4 records gobernando un scope = hot zone (warning al emitir).

## Trade-offs

- Coordinación blanda en el working tree: el enforcement cubre las tools
  edit/write de Devin; un agente que escriba vía `exec` con redirección
  elude el hook (aislamiento duro real = git worktrees, complementario).
- Overlap de globs es conservador por prefijo — puede sobre-reportar
  (`src/**/*.py` contra `src/ipa/x.md`); dirección segura, falsos
  positivos preferibles a pisadas.
- Permisos expiran por TTL; `permit.py prune` marca los huerfanos como
  `expired` para auditoría.

## Ejemplos locales

Smoke verificado 2026-09-23: acquire PW-20260923-01 sobre
`src/ipa/agentic/**` adjuntó 6 precautions automáticamente; una segunda
sesión fue rechazada por conflicto de scope; el hook bloqueó un edit
ajeno e inyectó los records gobernantes al editar dentro del scope
propio; `close --eks-draft` cerró el ciclo.

## Addendum 2026-09-23 — hardening del circuito (v0.2.0)

Enmienda los bullets de *Solución* que describen el enforcement y el cierre:

- **`acquire` es atómico.** El check-de-conflicto + asignación de id +
  escritura corren bajo un lockfile `O_CREAT|O_EXCL` (`.acquire.lock` en el
  directorio de permisos, stale a los 30 s, espera máxima 5 s). Sin esto,
  dos sesiones podían pasar el check a la vez y quedarse ambas con un scope
  exclusivo solapado — el caso exacto de 3-4 sesiones paralelas. Misma
  disciplina de lease que PAT-007.
- **El cierre ya no puede perder conocimiento en silencio.** `SessionEnd`
  sigue liberando el scope (dejarlo vivo bloquearía a las demás sesiones
  hasta el TTL), pero todo permiso que cierra sin `--eks-draft` deja un
  marcador `.unharvested-<session>.json`; el siguiente `SessionStart` lo
  reporta y lo borra. El closeout deja de ser opcional-para-la-memoria.
- **`Stop` bloquea una sola vez por sesión** (marcador `.stop-reminded-*` +
  guard `stop_hook_active`): pide el closeout sin entrar en loop. Se
  prefiere `decision: block` a `additionalContext` porque el soporte de
  contexto en `Stop`/`PostCompaction` no está documentado (sí lo está
  `PostCompaction` como re-inyección, que se mantiene).
- **Higiene de estado**: `SessionStart` poda los `.seen-*.json` de más de
  7 días.
- **Hot zones del reporte**: `eks_report` expone `hot_zones_overlap`
  (criterio de prefijo, idéntico al que decide las precautions de un
  `acquire`) además de `hot_zones` (glob exacto). Antes el reporte podía
  decir "0 hot zones" mientras un permiso recibía 11 precautions.
- Cobertura: `tests/test_permit_guard.py` (15 tests) cubre bloqueo,
  inyección una-vez-por-sesión, marcador unharvested, Stop-once y poda;
  `tests/test_eks.py` agrega concurrencia de `acquire` (8 sesiones → 1 gana).