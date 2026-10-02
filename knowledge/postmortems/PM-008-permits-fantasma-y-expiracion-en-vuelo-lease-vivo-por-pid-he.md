---
id: PM-008
category: postmortem
status: accepted
created: 2026-10-02
updated: 2026-10-02
author: agent
components: [eks]
tags: [permits, tooling, lease, reaper, port-riapp]
related: [PAT-009, PAT-007, PM-009, PM-010]
supersedes: null
superseded_by: null
affects: [tools/work_permits.py, scripts/hooks/permit_guard.py, scripts/cli/permit.py, tests/test_permit_guard.py, AGENTS.md]
evidence:
  - tools/work_permits.py
  - scripts/hooks/permit_guard.py
  - scripts/cli/permit.py
  - tests/test_permit_guard.py
  - AGENTS.md
author_model: SWE-2 (Devin, sesion pumped-pine)
trigger: permit:PW-20261002-02
---

# PM-008 — Permits fantasma y expiración en vuelo: lease vivo por pid, heartbeat del hook y zonas blandas (port RIAPP)

## Impacto

El sistema de permits se portó de IPA a RIAPP, donde 5 agentes en paralelo
expusieron tres fallos que en IPA (uso personal, mayormente una sesión)
estaban latentes:

- **Permit fantasma**: en IPA quedó `PW-20260925-06` con `status: active`
  una semana después de morir su sesión — `alive()` por TTL lo ocultaba del
  circuito de bloqueo pero el campo `status` quedaba stale hasta un
  `prune` manual que nadie corría. Confirmado en disco al hacer este port
  (2026-10-02).
- **Expiración en vuelo**: el TTL (7200s) con heartbeat manual implicaba
  que un batch nocturno (3-5h) podía perder su scope a mitad de trabajo —
  otra sesión podía adquirirlo mientras el batch corría.
- **Modos muertos**: `advisory` y `survey` existían en `PERMIT_TYPES`
  desde el origen pero nadie los usaba — sin caso de uso documentado ni
  enforcement, todo era `exclusive`, incluso para docs.

## Causa raíz

El lease solo tenía una señal de muerte (TTL) y el heartbeat era un comando
manual que nadie invocaba. Sin señal de liveness del holder no hay forma de
distinguir "sesión trabajando" de "proceso muerto hace días". Mismo
hallazgo que `vram_lock` (PAT-007), re-aprendido en el dominio de permits.

## Corrección (portada de RIAPP PM-009, aprobada por el CTO allá el 2026-09-29)

1. **Reaper por pid**: el permit guarda el `pid` del proceso holder
   (`os.getppid()` del acquire) y `pid_alive` lo sondea con caché de 2s —
   Windows vía `ctypes` (`OpenProcess`/`GetExitCodeProcess`, STILL_ACTIVE=259),
   POSIX vía `os.kill(pid, 0)`. Ante fallo de sonda, el TTL manda (la sonda
   no mata de más).
2. **Lease vivo**: holder con pid vivo manda sobre el TTL — un batch >ttl
   ya no caduca en vuelo. Permits viejos sin `pid` conservan la semántica
   TTL pura (backward compat testeado).
3. **Heartbeat automático**: `_heartbeat_mine` en el hook renueva el lease
   de los permits de la sesión en cada tool call.
4. **Zonas blandas**: `docs/` y `knowledge/` bajo `exclusive` ajeno avisan
   pero no bloquean (`SOFT_ZONE_PREFIXES`) — el STOP solo donde el riesgo
   real es de código. Esto activa de verdad los modos `advisory`/`survey`.
5. **Hints de granularidad**: el mensaje de bloqueo/conflicto sugiere
   scope a nivel de archivo; `permit list` anota `alive` por permit.

Verificado: `tests/test_permit_guard.py` +7 tests (soft zones, heartbeat,
live-holder-survives-ttl, dead-pid, legacy-sin-pid) — 64 tests verdes en
los archivos tocados.

**Nota de entorno (descubierto al portar)**: en Devin Desktop cada comando
corre bajo un wrapper transitorio — `os.getppid()` en `acquire` captura un
launcher que muere al terminar el comando, así que el permit se auto-reapea
al instante (PW-20261002-01/02/03 lo demostraron). En el runtime de RIAPP
el parent era un shell de sesión longevo. Workaround operativo: editar el
permit a `pid: null` → cae al modo TTL+heartbeat del hook, que con
heartbeats por tool call cubre la sesión correctamente. Si se quiere
liveness real acá hace falta otra señal (la sesión es lógica, no un
proceso).

## Prevención

Tests de lease en `test_permit_guard.py`; modos documentados con caso de
uso en AGENTS.md y `.devin/rules/eks-workflow.md`. `permit.py list`
muestra `alive` para inspección.

## Lección reutilizable

1. El valor del sistema no era el lock sino la **inyección de contexto
   gobernante** — al flexibilizar (zonas blandas) se afloja el lock, no la
   inyección.
2. Un modo sin caso de uso documentado ni enforcement es un modo que no
   existe.
3. La liveness de un lease necesita la señal del holder (pid), no solo un
   TTL — PAT-007 ya lo sabía en runtime; el dominio dev-time lo
   re-aprendió.
4. Los bugs de coordinación no aparecen con un solo agente: el port a un
   repo multi-sesión fue el estrés que los reveló. Traer los postmortems
   junto con el código preserva el porqué.
