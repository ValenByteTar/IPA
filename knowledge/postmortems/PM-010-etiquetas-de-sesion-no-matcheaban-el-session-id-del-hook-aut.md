---
id: PM-010
category: postmortem
status: accepted
created: 2026-10-02
updated: 2026-10-02
author: agent
components: [eks]
tags: [permits, tooling, session-id, port-riapp]
related: [PM-008, PAT-009]
supersedes: null
superseded_by: null
affects: [scripts/hooks/permit_guard.py, tests/test_permit_guard.py, AGENTS.md, .devin/rules/eks-workflow.md]
evidence:
  - scripts/hooks/permit_guard.py
  - tests/test_permit_guard.py
  - AGENTS.md
author_model: SWE-2 (Devin, sesion pumped-pine)
trigger: permit:PW-20261002-02
---

# PM-010 — Etiquetas de sesión no matcheaban el session_id del hook: auto-bloqueo del holder (port RIAPP)

## Impacto

En RIAPP (PM-002 allá), agentes que adquirían su permit con una etiqueta
legible (`"A"`, `"V1"`, `devin-mobile-push-20260925`) quedaban **bloqueados
por su propio permit exclusive**: el guard compara el `session_id` que
Devin pone en el payload del hook contra `permit.session`, y la etiqueta
nunca coincidía.

En IPA ocurrió exactamente lo mismo: `PW-20260925-04` se cerró a los 78
segundos con nota `"reissue: usar session id real del hook
(star-caravel)"` — el holder tuvo que descubrir el id real a mano y
re-adquirir (PW-20260925-05). Esta sesión lo verificó de nuevo: descubrir
el propio session_id requirió inspeccionar `.seen-*.json` en
`outputs/devin/permits/`, porque nada se lo decía al agente.

## Causa raíz

El contrato "el valor de `--session` debe ser el session_id del runtime"
existía pero era invisible: el hook conocía el id en cada evento y nunca se
lo comunicaba a la sesión. Documentarlo en AGENTS.md no bastaba — el agente
no tiene forma de adivinarlo.

## Corrección

- `on_session_start` inyecta `"Your Devin session_id: <id> — pass it as
  --session in permit.py acquire"` en el contexto inicial.
- `on_post_compaction` re-inyecta el id siempre (antes solo lo hacía si
  había permits propios — una compactación temprana perdía el dato).
- AGENTS.md y `.devin/rules/eks-workflow.md` documentan que `<id>` es el
  session_id del hook.

## Prevención

`test_session_start_injects_session_id_for_self_identification` y
`test_post_compaction_reminds_session_id_without_permits` en
`tests/test_permit_guard.py`.

## Lección reutilizable

Un identificador que el runtime conoce y el agente necesita debe ser
**inyectado**, no documentado. Si el descubrimiento de un valor requiere
leer estado interno del sistema, la interfaz está mal — no el usuario.
