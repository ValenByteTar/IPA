---
id: DEC-011
category: decision
status: proposed
created: 2026-09-25
updated: 2026-09-25
author: agent
components: [notifications, dashboard, configuration]
tags: [push, movil, android, mtls, long-poll, slot-unico, seguridad, research]
related: [DEC-002, DEC-005, PAT-009]
supersedes: null
superseded_by: null
affects: [src/ipa/notifications/**, configs/notifications.yaml, scripts/cli/notifications.py, mobile/**, outputs/notifications/**]
evidence: [docs/plans/mobile-push-notifications.md, tests/test_notifications_store.py, tests/test_notifications_tls.py, tests/test_notifications_service.py, tests/test_notifications_hook.py, tests/test_notifications_cli.py]
author_model: GLM-5.3 Flash Max (Devin)
trigger: permit:PW-20260925-09
---

# DEC-011 — IPA Push: notificaciones propias al movil (mTLS long-poll, slot unico)

## Contexto

El usuario quiere que IPA le avise en su teléfono Android ("investigación lista"
como primer evento). No existe ningún canal de notificación externa: todo queda
en logs y el dashboard local. Las opciones de mercado (ntfy, Telegram, Pushover)
fueron descartadas explícitamente por el usuario: quiere un sistema **propio,
diseñado desde 0**, sin terceros en el circuito. Restricciones aprobadas: sin
Tailscale ni puertos abiertos en la fase inicial (alcance: LAN), app Android
nativa en Kotlin (no PWA), y seguridad como prioridad — "que no genere una
apertura en ninguno de los 2 sistemas".

Alternativas evaluadas y descartadas (cementerio):

- **ntfy / Telegram / Pushover**: terceros en el circuito (descarte del usuario).
- **PWA + Web Push (VAPID)**: exige HTTPS con CA instalada en el móvil y delega
  la entrega al push service del navegador (tercero de facto).
- **Token portador**: si se filtra de la app, entra cualquiera que lo tenga.
- **MAC/IP binding**: suplantables en segundos; Android 10+ aleatoriza la MAC
  por red y las apps ya no pueden leer la suya; frágil ante rotación. El usuario
  lo propuso y lo descartó al ver el análisis.
- **WebSocket propio (RFC 6455 a mano)**: más superficie de parser propio (los
  bugs de frames son la fuente clásica de vulns en servidores WS caseros), con
  la misma latencia efectiva que long-poll. Queda como posible v2.

## Decisión

Sistema push propio en dos mitades (Fase 0 servidor, Fase 1 app):

1. **Bounded context nuevo** `ipa/notifications/` — separado del dashboard y del
   agent core. El dashboard lo hospeda como un thread más (control room, no
   segundo runtime) y solo llama a `notify()`; no conoce el protocolo.
2. **Transporte**: HTTP long-poll sobre TLS con **mTLS** — CA propia del PC
   (`outputs/notifications/tls/`), server cert con SAN = IP LAN (regenerable si
   la IP cambia; la app fija la huella de la CA, no el hostname), un cert
   cliente para el teléfono entregado como `.p12` en el emparejamiento. Sin
   token en el protocolo: la identidad es posesión de la clave privada.
3. **Slot único**: `devices` tiene exactamente una fila. El fingerprint sha256
   del cert cliente debe coincidir con el del slot (403 `slot_occupied` si no).
   Emparejar exige ejecutar `notifications.py pair` en el PC (`--replace`
   explícito para desalojar); `revoke` libera y revoca. Robar el `.p12` no da
   la plaza si se rotó; tomar la plaza exige acceso al PC.
4. **Protocolo v1 congelado**: `GET /health`, `GET /poll?since=&timeout=`
   (long-poll, retorna al instante con eventos nuevos), `POST /ack`. Backlog
   por `last_ack_id` — nada se pierde caído el enlace. Colapso anti-spam por
   `(event_type, collapse_key)` (para research: la query). Límites duros:
   ≤5 conexiones, ≤1 poller por fingerprint, body ≤64 KB, poll ≤60 s.
5. **Frontera de contenido (DEC-002)**: los avisos viajan como SOLO metadatos
   del evento (query, estado, conteo, corpus) — nada de sesiones/episodios
   personales sale del equipo.
6. **Hook**: el watcher de research del dashboard (patrón `RESEARCH_WATCH`,
   DEC-005) llama `notify_research()` en la transición done/failed,
   independiente de `session_id` y `CHAT_BUSY` (research por CLI también
   avisa), best-effort: un fallo de push NUNCA rompe el watcher.
7. **Estado**: `outputs/notifications/push.db` (SQLite WAL, derivable y
   reconstruible) + material TLS en `outputs/notifications/tls/` (cubierto por
   `outputs/*` en `.gitignore` — sin cambios en `.gitignore`).
8. **Dependencias**: ninguna nueva. Certificados con `cryptography`, ya
   presente en el venv como transitiva de google-auth/pdfminer.six. Servidor
   HTTP con stdlib (`ThreadingHTTPServer` + `ssl`, mismo patrón del dashboard).

Queda fuera: entrega con datos móviles (Fase 2: Tailscale/port-forward), más
eventos (errores, pipeline, digest), WebSocket v2, QR de emparejamiento.

## Consecuencias

- Avisos 100% propios: sin terceros, sin cuentas, sin nube; el único listener
  LAN nuevo del proyecto, con firewall con scope (perfil privado + subred).
- Primer evento "investigación lista" funciona igual para research desde chat
  que por CLI; los avisos encolados sobreviven reinicios del dashboard.
- Coste: primera superficie LAN del proyecto — el modelo de acceso (mTLS +
  slot único + firewall + límites + auditoría de rechazos) es la mitigación y
  queda documentado aquí. TLS sin hostname check del lado app (pinning de CA).
- Mantenimiento: material TLS con vigencias (CA 10y, server 5y, client 5y);
  `ensure_server_cert` regenera el server cert si cambia la IP.
- Gotcha Windows (encontrado en smoke real): `SO_REUSEADDR` permite doble-bind
  silencioso del mismo puerto (dos procesos compartiendo 8766). El server
  nace con `allow_reuse_address=False` — el segundo bind falla ruidoso y
  `start_service_thread` reintenta (TIME_WAIT tras restart del dashboard).
- Rollback: `push.enabled: false` en config apaga el servicio; borrar
  `outputs/notifications/` resetea cola y material TLS (re-emparejar).

## Evidencia

- Plan completo: `docs/plans/mobile-push-notifications.md`.
- Tests (sockets y TLS reales en localhost): `tests/test_notifications_store.py`
  (9: cola, slot único, colapso, reintentos, retención),
  `tests/test_notifications_tls.py` (7: CA idempotente, SAN/IP, .p12,
  revocación), `tests/test_notifications_service.py` (8 end-to-end: handshake
  sin cert falla, fingerprint ajeno → 403 slot_occupied, revocado → 403,
  not_paired → 403, ciclo poll→ack→backlog, wake, límites, body oversize),
  `tests/test_notifications_hook.py` (5: metadatos only, colapso por query),
  `tests/test_notifications_cli.py` (2: flujo pair/replace/revoke real).
- Implementación: `src/ipa/notifications/` (store, tls_material, transport,
  service, config), `scripts/cli/notifications.py`, hook + arranque en
  `src/ipa/dashboard/server.py`, `configs/notifications.yaml`.

## Alcance

Fronteras tocadas: el dashboard gana un thread de servicio pero no gana lógica
de dominio de notificaciones (vive en `ipa/notifications/`); el agent core no
cambia; el contenido que sale del equipo queda restringido a metadatos de
evento (refuerza DEC-002). Evolución por supersede: cambiar transporte
(WebSocket v2), modelo multi-dispositivo o añadir eventos = nuevo DEC que
supersede este. Fase 1 (app `mobile/`) y Fase 2 (datos móviles) se rigen por
el plan `docs/plans/mobile-push-notifications.md`.
