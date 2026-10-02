# Plan — IPA Push: notificaciones al móvil (Android)

Objetivo: sistema de notificaciones **propio, de 0 a 100** — un servicio push en el
PC y una app Android nativa en Kotlin — para que IPA avise al teléfono del usuario.
Primer evento: **investigación lista** (`research_topic` terminada, ok o fallida).

Invariantes que se respetan: el dashboard sigue siendo control room, no segundo
runtime (docs/architecture/dashboard.md); el contenido personal nunca sale del
equipo (DEC-002); estado derivado y reconstruible bajo `outputs/`; sin
dependencias nuevas en el servidor (`cryptography` ya presente como transitiva
+ stdlib); sin tocar `.gitignore`.

---

## Decisiones aprobadas por el usuario (sesión 2026-09-25)

1. **Sistema propio desde 0** — descartados ntfy (self-hosted y público), Telegram
   y Pushover por ser terceros en el circuito.
2. **App Android nativa en Kotlin** (no PWA/Web Push) — elegido explícitamente.
3. **Primer evento**: investigación lista. Sin Tailscale ni puertos abiertos en la
   fase inicial → alcance de entrega: **misma Wi-Fi (LAN)**.
4. **Transporte**: long-poll HTTP sobre TLS. Elección del agente con criterio de
   seguridad: menos código propio de parsing que un WebSocket RFC 6455 hecho a
   mano (los bugs de parser de frames son la superficie clásica), misma latencia
   efectiva (el poll queda colgado en el servidor y retorna al instante). WS queda
   como posible v2 detrás de la misma interfaz de transporte.
5. **Seguridad**: **mTLS** (certificados mutuos con CA propia del PC) + **slot
   único** + emparejamiento solo desde el PC + firewall con scope LAN.
   Descartados y por qué (cementerio de alternativas evaluadas):
   - *Token portador*: si se filtra de la app, entra cualquiera que lo tenga.
   - *MAC/IP binding*: suplantables en segundos; Android 10+ aleatoriza la MAC
     por red y las apps ya no pueden leer la suya; frágil ante rotación.
   - *PWA + Web Push*: requiere HTTPS con CA instalada en el móvil y delega la
     entrega al push service del navegador (tercero).
6. **La app vive en `mobile/`** de este repo (proyecto Gradle propio).
7. **Sin token en el protocolo**: mTLS lo reemplaza — la identidad es posesión de
   la clave privada del certificado, que nunca sale del teléfono.

## Arquitectura

Bounded context nuevo, separado del dashboard y del agent core:

```text
src/ipa/notifications/
  __init__.py     API pública: notify(), notify_research(), PushService
  store.py        Cola + slot de dispositivo (SQLite, outputs/notifications/push.db)
  tls_material.py CA propia, server cert, client cert (.p12)
  transport.py    HTTP long-poll sobre TLS (ThreadingHTTPServer + ssl)
  service.py      Bucle del servicio: bind, accept, auth por fingerprint, poll, ack
  config.py       configs/notifications.yaml + resolución de IP LAN
scripts/cli/notifications.py   init-ca | pair | revoke | serve | send | devices | status
mobile/                        App Android (Fase 1, proyecto Gradle propio)
```

El dashboard hospeda el servicio como un thread más (arranque en `server.py`);
el estado vive en `outputs/notifications/` (reconstruible). El dashboard no
conoce el protocolo: solo llama a `notify()`.

## Modelo de seguridad

Amenaza modelada: otro dispositivo dentro de la LAN doméstica (invitado, IoT
comprometido, vecino con la contraseña Wi-Fi). Nada se expone a internet: sin
port forwarding, sin UPnP, sin Tailscale en fase inicial.

Capas (todas requeridas):

1. **mTLS**: el handshake TLS exige certificado de cliente firmado por nuestra CA
   (`ssl.CERT_REQUIRED`). Sin cert válido, la conexión muere en el handshake —
   no hay ruta HTTP que atacar. Nada de secretos viaja por el cable.
2. **Slot único**: `devices` tiene exactamente una fila. El servidor compara el
   fingerprint sha256 del cert cliente contra el del slot; cert CA-válido con
   fingerprint distinto → `403 slot_occupied` + log.
3. **Emparejamiento solo desde el PC**: `notifications.py pair` genera el
   certificado de cliente y el `.p12` (contraseña de un solo uso, impresa una
   vez). Tomar la plaza exige ejecutar un comando en el PC; `--replace` es
   explícito para desalojar un slot ocupado. `revoke` borra slot y añade el
   fingerprint a la lista de revocados.
4. **Firewall Windows con scope**: regla solo perfil *Privado* y solo la subred
   LAN (ej. `192.168.1.0/24`). Comando en Fase 0 (requiere admin, una vez).
5. **Límites duros**: ≤5 conexiones simultáneas, ≤1 por fingerprint, body ≤64 KB,
   poll timeout ≤60 s, read timeout 90 s. Anti-exhaustión/DoS local.
6. **Auditoría**: toda conexión rechazada (handshake, fingerprint, límite) queda
   en el log del servicio con IP y motivo.

Material TLS en `outputs/notifications/tls/` (`ca.key`, `ca.crt`, `server.crt`,
`server.key`, `client-<name>.p12`, `revoked.json`). Cubierto por `outputs/*` en
`.gitignore` — no se modifica `.gitignore`.

**Lado teléfono**: la app es solo cliente — conexiones salientes al PC, ningún
listener en el móvil, permisos mínimos (`INTERNET`, `POST_NOTIFICATIONS`,
`FOREGROUND_SERVICE`).

## Protocolo v1 (JSON sobre HTTPS long-poll)

Rutas (todas con mTLS + fingerprint del slot):

```
GET  /api/push/v1/health              → 200 {"ok":true,"server":"ipa-push/0.1"}
GET  /api/push/v1/poll?since=<id>&timeout=<s≤60>
     → retorna al instante si hay eventos con id>since;
       si no, espera hasta timeout → {"events":[...],"last_id":N} | {"events":[],"last_id":N}
POST /api/push/v1/ack  {"ids":[...]}  → {"acked":N}
```

Errores: `403 {"error":"slot_occupied"}`, `429 {"error":"too_many_connections"}`,
`400` malformed. La app persiste `last_ack_id`; al reconectar poll con
`since=<last_ack_id>` → **backlog automático** (nada se pierde caído el enlace).

Eventos v1: `research.done`, `research.failed`.

```json
{"id":42,"event_type":"research.done","title":"Investigación lista: <query>",
 "body":"12 URLs · corpus agent_research","payload":{"query":"...","status":"ok",
 "urls_ingested":12,"corpus":"...","finished_at":"2026-09-25T10:00:00Z"}}
```

Contenido: **solo metadatos del evento** (DEC-002 — nada de sesiones/episodios
viaja al móvil). Colapso anti-spam: si hay entregas `pending` del mismo
`(event_type, clave de colapso)` (para `research.done`: la query), se marcan
`superseded` y solo viaja la última.

## Esquema de datos — `outputs/notifications/push.db`

```sql
devices(
  id TEXT PRIMARY KEY CHECK (id = 'slot'),
  cert_fingerprint TEXT NOT NULL,        -- sha256 del DER del cert cliente
  name TEXT, paired_at TEXT, last_seen_at TEXT, last_ack_id INTEGER DEFAULT 0
)
notifications(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT, title TEXT, body TEXT, payload_json TEXT,
  created_at TEXT, priority INTEGER DEFAULT 1
)
deliveries(
  notification_id INTEGER, device_id TEXT,
  state TEXT CHECK (state IN ('pending','sent','acked','superseded','failed')),
  attempts INTEGER DEFAULT 0, sent_at TEXT, acked_at TEXT, last_error TEXT,
  PRIMARY KEY (notification_id, device_id)
)
```

Reintentos: `pending`/`sent` sin ack se reenvían al reconectar; `attempts` ≥5 →
`failed` (visible en `status`). Retención: `acked`/`superseded` > 7 días se
purgan (config). La DB es derivable: borrarla solo pierde historial de avisos.

## Configuración — `configs/notifications.yaml`

```yaml
push:
  enabled: true
  bind_ip: auto        # resuelve la IP LAN al arranque; o fija, ej. 192.168.1.50
  port: 8766
  max_connections: 5
  poll_timeout_s: 25
  max_backlog: 50
  retention_days: 7
events:
  research: true       # research.done / research.failed
```

Sin secretos en el YAML: todo el material TLS vive en `outputs/notifications/tls/`.

## Hook — investigación lista

Punto de integración: el watcher del dashboard que ya consume
`outputs/web_dashboard/research_progress.json` y notifica a la sesión
(patrón `RESEARCH_WATCH`, DEC-005). En la transición a `done`/`failed`:

```python
from ipa.notifications import notify_research
try:
    notify_research(progress)          # arma título/body/payload y encola
except Exception:
    ...  # log a TraceLog — un fallo de push NUNCA rompe el watcher
```

`notify()` es best-effort por diseño: si el servicio no está corriendo, la
notificación queda `pending` en la DB y se entrega al reconectar el dispositivo.

---

## Fase 0 — Servidor (este repo)

**Work permit** (PAT-009): scope `src/ipa/notifications/**`,
`src/ipa/dashboard/**`, `configs/**`, `scripts/cli/**`, `tests/**`,
`docs/plans/**`. Verificar `permit.py check` antes de editar. Nota de entorno
(2026-09-25): los hooks llegan sin `session_id` (payload vacío) — un permit
exclusive se auto-bloquea; usar `--type advisory` (patrón establecido hoy,
ver close notes de PW-20260925-02/03) salvo coordinación real entre sesiones.

1. **`src/ipa/notifications/store.py`** — schema + migración idempotente,
   `enqueue()`, `fanout()`, `next_batch(since)`, `ack()`, `supersede()`,
   `prune()`, `bind_slot()`, `revoke()`, `slot_fingerprint()`.
2. **`src/ipa/notifications/tls_material.py`** — `init_ca()` (idempotente),
   `ensure_server_cert(ip)` (SAN = IP LAN, regenerable), `issue_client(name)`
   → `.p12` + huella, `revoke_fingerprint()`. **Resuelto**: generación con
   `cryptography` (ya transitiva de google-auth/pdfminer.six en el venv —
   sin deps nuevas).
3. **`src/ipa/notifications/transport.py`** — `ThreadingHTTPServer` + `ssl`
   (`CERT_REQUIRED`), rutas del protocolo, límites, logging de rechazos.
4. **`src/ipa/notifications/service.py`** — arranque/parada limpia, bind IP
   (`auto` = resolución LAN), integración con store.
5. **`src/ipa/notifications/config.py`** + `configs/notifications.yaml`.
6. **`scripts/cli/notifications.py`** — `init-ca | pair [--replace] | revoke |
   serve | send | devices | status`. `pair` imprime: URL del servidor, huella
   CA (para la app), ruta del `.p12`, contraseña de un solo uso.
7. **Hook en el watcher de research** (`src/ipa/dashboard/server.py`) —
   transición done/failed → `notify_research()` envuelta en try/except.
8. **Arranque del servicio** en `server.py` (thread, `enabled: true`).
9. **Firewall** (una vez, admin): `netsh advfirewall firewall add rule
   name="IPA Push" dir=in action=allow protocol=TCP localport=8766
   profile=private remoteip=<subred LAN del usuario>`.
10. **Verificar puerto 8766 libre** (`netstat -ano | findstr 8766`).

**Tests** (`tests/test_notifications_*.py`, sockets y TLS reales en localhost):

- `store`: enqueue/fanout/ack/supersede/prune/bind/revoke; slot único.
- `tls_material`: init-ca idempotente, pair/replace/revoke, huellas.
- `transport`+`service` end-to-end: cliente Python de test con cert → poll
  inmediato, poll timeout, ack, backlog tras reconexión, handshake sin cert
  falla, cert CA-válido con fingerprint ajeno → 403, límites (429, 400).
- `hook`: progress file fake done/failed → notify disparada; excepción de push
  no rompe el watcher.
- `cli`: pair imprime material completo; revoke limpia slot.

**Criterios de aceptación**: suite pytest completa en verde; `serve` + `send`
recibidos y ackeados por el cliente de prueba; `validate_eks.py` y
`eks_report.py` sin errores; DEC-011 → `accepted` con `evidence:` existente.

## Fase 1 — App Android (`mobile/`)

**Toolchain (guía al usuario)**: Android Studio (incluye JDK 17 + SDK),
Kotlin, Gradle wrapper. minSdk 26, targetSdk 35. Sin Play Services.

1. Proyecto Gradle en `mobile/` (`settings.gradle.kts`, `app/`), build outputs
   ignorados por el propio `.gitignore` del proyecto móvil.
2. `MainActivity`: estado del enlace + settings (URL del servidor, huella CA,
   selección del cert vía `KeyChain.choosePrivateKeyAlias` — aprobación única).
3. `PushService` (foreground service, tipo `dataSync` — Android 14 exige
   declararlo): bucle long-poll con OkHttp + `SSLContext` (CA fijada por
   huella + cert cliente de KeyChain), backoff 1→60 s, ack inmediato,
   `last_ack_id` persistido (SharedPreferences).
4. Canales de notificación: `research` (importancia alta) y `service`
   (la notificación persistente, silenciosa).
5. Permisos mínimos: `INTERNET`, `POST_NOTIFICATIONS` (runtime),
   `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_DATA_SYNC`.
6. Instalación: `adb install` o copia directa del APK (origen desconocido).
7. Batería: sin restricciones para la app + guía por fabricante
   (autobinicio Xiaomi/Samsung, etc.).
8. **Protocolo de pruebas manual**: aviso de prueba desde el PC → notificación
   en el teléfono; Wi-Fi off/on → reconexión + backlog; dashboard reiniciado
   (watchdog) → reconexión; teléfono fuera de la Wi-Fi → sin entrega (esperado,
   fase LAN).
9. **EXP-010** (EKS): latencia notify→notificación (objetivo <2 s en LAN),
   tiempo de reconexión, backlog tras reconexión, consumo de batería 24 h,
   fiabilidad del foreground service en el teléfono real del usuario.

**Criterios de aceptación**: aviso de prueba llega al teléfono en <2 s;
reconexión automática con backlog; EXP-010 registrado.

## Fase 2 — Ampliación posterior (no en este plan de ejecución)

- Entrega con datos móviles: Tailscale (preferido) o port forwarding + DDNS.
- Más eventos: errores/watchdog, pipeline/ingesta, reporter, digest periódico.
- WebSocket como v2 del transporte (misma interfaz, store compartido).
- QR de emparejamiento (URL + huella CA) para evitar tipeo.
- Acciones en la notificación (abrir resultado en el dashboard, si Fase 2
  añade acceso remoto).

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Batería/fabricante mata el foreground service | whitelist + guía por marca; EXP-010 lo mide |
| IP LAN cambia (DHCP) | `bind_ip: auto`; la app fija la CA (no el hostname) → solo cambia la URL en la app |
| Dashboard se reinicia (watchdog) | backlog por `last_ack_id`; reconexión con backoff |
| `.p12` filtrado | poseer el cert = identidad (como llave): `revoke` + `pair` nuevo; el slot exige fingerprint |
| Doze/App Standby pausa el poll | foreground service + whitelist; medido en EXP-010 |
| Puerto 8766 ocupado | verificación en Fase 0; puerto configurable |
| Scope creep del protocolo | v1 congelado: 3 rutas, 2 eventos; extensiones en Fase 2 |

## Huecos de conocimiento

- Fiabilidad del foreground service en el teléfono concreto del usuario
  (marca/modelo) — se valida en Fase 1 (EXP-010).
- Generación de certificados: `openssl` CLI vs `cryptography` — decidir en
  implementación según lo ya presente en el entorno (sin deps nuevas).
- Comportamiento real de Doze sobre long-poll en el dispositivo — EXP-010.

## EKS

- **DEC-011** (nueva, `proposed` → `accepted` al cerrar Fase 0): sistema push
  propio; transporte long-poll HTTPS mTLS; slot único + emparejamiento desde el
  PC; frontera de contenido (solo metadatos de evento, DEC-002); estado en
  `outputs/notifications/`. `affects: ["src/ipa/notifications/**",
  "configs/notifications.yaml", "scripts/cli/notifications.py", "mobile/**",
  "outputs/notifications/**"]`. `trigger: permit:PW-*`.
- **EXP-010** (Fase 1): mediciones de latencia/fiabilidad/batería.
- Permits por fase según scope (Fase 0 arriba; Fase 1: `mobile/**`).

## Convenciones y limpieza respetadas

- `outputs/notifications/` es derivable/reconstruible (borrarla solo pierde
  historial de avisos). No se toca `Landing/`, `Archive/`, `Transit/`.
- Sin cambios en `.gitignore` (el material TLS queda cubierto por `outputs/*`).
- Sin dependencias nuevas en `requirements.txt` para el servidor.
- Commits sin atribución Devin (regla absoluta del usuario).
