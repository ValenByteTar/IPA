# Plan preliminar — conectividad de correo (IPA Mail)

> Estado: **preliminar / aparcado**. Documento de exploración escrito el
> 2026-09-25. No hay permiso de trabajo ni código asociado. Mover a
> `docs/plans/` cuando se libere el permiso exclusivo sobre ese directorio
> (PW-20260925-04) o cuando el plan se active.

## Motivación honesta

La pregunta que decide si esto vale la pena: **¿el correo contiene conocimiento
que se querría buscable y que el scraper no puede obtener?**

Análisis de los casos de uso, ordenados por fuerza:

| Caso | Veredicto |
|---|---|
| Ingesta de suscripciones curadas (newsletters, listas arXiv, changelogs, advisories) | **Fuerte** — extiende el intelligence-gathering del scraper con señal que no está en la web abierta |
| Salida SMTP para digests largos del Reporter | Marginal — push cubre alertas; email solo gana en reportes largos |
| Canal de entrada (mandar consultas por mail) | Débil — chat CLI / dashboard / push son estrictamente mejores |
| Ingesta del inbox completo | **Negativo** — 90% ruido (recibos, marketing, notificaciones); contamina embeddings y degrada retrieval |
| Triage agéntico / extracción de compromisos | Off-mission — convierte a IPA en asistente personal; decisión de producto aparte |

Conclusión: el único caso que encaja con la misión del sistema es **email como
adaptador de adquisición de fuentes curadas**. Todo lo demás se descarta o se
difere explícitamente.

## Diseño general

Email entra por el mismo patrón que el scraper web: un fetcher deposita
artefactos en `Landing/mail/<cuenta>/`, el pipeline Landing → Transit → Archive
hace el resto (parse, chunk, index). Nada de acceso "en vivo" al buzón en v1 —
provenance y trazabilidad vienen gratis por la ruta de artefactos.

```
IMAP ──poll──> mail_fetcher ──.eml──> Landing/mail/<cuenta>/
                                        │
                            mime_router (message/rfc822)
                                        │
                            eml_parser: body→txt, adjuntos→Landing
                                        │
                            pipeline normal (chunk → embed → index)
```

Dedup por UID+UIDVALIDITY de IMAP en una history DB. **La DB no vive bajo
`Landing/`** — la regla de cleanup solo permite `Landing/web/scrape_history.db`
ahí; usar `outputs/agent/mail_history.db` (o equivalente).

## Fases

### Fase 0 — Experimento mínimo: parser `.eml` + drop manual

Validar que el contenido de correo aporta valor antes de construir el fetcher.

- `src/ipa/ingestion/eml_parser.py`: stdlib `email` package. Extrae cuerpo
  (prefiere `text/plain`; fallback `text/html` → texto vía trafilatura o el
  extractor HTML existente), escribe `.txt` + metadatos (from/subject/date) en
  cabecera del artefacto. Adjuntos se re-inyectan en `Landing/` para routing
  recursivo por `mime_router`.
- `mime_router.py`: añadir `message/rfc822` (`.eml`) → parser `eml`. Sin magic
  bytes confiables — detección por extensión + heurística de headers
  (`From:`, `Subject:`, `MIME-Version:`).
- Flujo manual: exportar/arrastrar `.eml` a `Landing/` durante ~2 semanas.
- **Criterio de continuar**: si el drop manual se usa y el contenido aparece
  útil en búsquedas reales → Fase 1. Si no, el plan muere aquí sin costo.
- Tests: fixtures `.eml` sintéticos (multipart, adjuntos, html-only) en
  `tests/`; validar extracción real, no existencia de archivos.

### Fase 1 — IMAP poller (solo-lectura, scope acotado)

Solo si Fase 0 demostró uso.

- `src/ipa/acquisition/mail_fetcher.py`: espejo estructural de
  `web_scraper.py`. `imaplib` (stdlib, cero deps). Poll periódico, fetch por
  UID, marca `\Seen` opcional.
- Config `configs/mail.yaml` (modelo `scrape_sites.yaml`): cuentas, host/puerto
  IMAP, carpeta/label acotado (ej. `ipa-ingest`, **no** INBOX completo),
  allowlist de remitentes/asuntos, `days_back`, `max_fetch`.
- Credenciales: app-passwords en env vars (`IPA_MAIL_*`), nunca en config
  commiteado. OAuth2 queda fuera de v1.
- Dedup: UID+UIDVALIDITY en `outputs/agent/mail_history.db` (mismo patrón de
  jobs/claims que `ScrapeHistory` si se quiere retry durable).
- Scheduling: Tier task del `idle_scheduler` (leases/locks ya existen) o
  invocación manual vía CLI tipo `run_web_scrape.py`.
- Filtro de ruido: allowlist de senders/carpetas es obligatoria, no opcional —
  es lo que separa "fuente curada" de "inbox".

### Fase 2 — SMTP out (opcional, independiente)

- `smtplib` para entregar reportes/digests del Reporter por correo.
- Solo tiene sentido si el canal push resulta insuficiente para contenido
  largo. Re-evaluar cuando push esté en producción.

### Fase 3 — Tools agénticas (diferido, requiere decisión aparte)

- `mail.search` / `mail.read` en el registry unificado (DEC-005), posiblemente
  vía MCP de correo externo como complemento de acceso vivo.
- Cualquier acción de escritura (`send`, `reply`, `archive`) va detrás de gate
  de aprobación explícito. Leer es barato; escribir en nombre del usuario no.
- Fuera de scope de este plan salvo que IPA pivote a asistente personal.

## Riesgos y mitigaciones

- **Prompt injection**: el correo entrante es untrusted. Cuando texto de un
  mail llegue al LLM se trata como dato, nunca como instrucción. Reusar la
  postura de `content_safety` (untrusted download → artefacto).
- **Ruido en el índice**: mitigado por allowlists + label acotado. Si el
  retrieval se degrada, el fix es apretar filtros, no añadir capacidad.
- **Credenciales**: app-passwords por cuenta, env vars, scope read-only de
  IMAP (sin `DELETE`, sin mover mensajes en v1).
- **Duplicación con scraper**: si una newsletter también existe en la web, la
  dedup de contenido del pipeline ya lo absorbe (corpus_dedupe).

## Decisiones abiertas

- ¿Qué suscripciones concretas se quieren ingeribles? (define la allowlist de
  Fase 1 y si el experimento tiene sentido)
- ¿`.eml` como artefacto persistente en Archive, o solo el `.txt` derivado?
  (afecta cuánto queda en Archive vs reproducible)
- ¿Gmail API / MS Graph en vez de IMAP genérico? (push real, labels nativos —
  más setup; solo si la latencia del polling resulta problema)
- ¿El drop manual de Fase 0 incluye también `.msg` de Outlook? (requeriría
  `extract-msg`, una dep extra — posponer salvo necesidad real)

## Fuera de scope (registrado para que no reaparezca)

- Envío/respuesta de correos por el agente sin gate humano.
- Sincronización de calendario, contactos, tareas.
- Acceso MCP de correo como sustituto de la ingesta (puede coexistir en Fase 3
  como acceso vivo, pero no reemplaza la ruta de artefactos).
