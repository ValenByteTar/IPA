---
description: "Checkpoints obligatorios en adquisiciones largas de datos — novedad, no volumen"
trigger: always_on
---

# Checkpoints obligatorios en adquisiciones largas de datos

(Portado de RIAPP, donde nació de un postmortem real: un downloader acumuló
~800k líneas que eran copias de la misma página porque el servidor ignoraba
el parámetro de paginación. En IPA aplica a `run_web_scrape`, corridas de
benchmarks largas y cualquier ingesta resumible.)

Cuando una descarga/ingesta es **larga y resumible**, parar a inspeccionar
es gratis: el estado guardado (`scrape_history.db`, manifest append-safe)
permite reanudar sin pérdida. Por eso los checkpoints de contenido son
obligatorios, no opcionales.

## Protocolo

1. **Pre-flight (antes de lanzar):** pedir las dos primeras unidades de
   trabajo (dos URLs, dos documentos) y verificar que traen **contenido
   distinto**. 30 segundos detectan paginación rota, endpoints que ignoran
   parámetros, o respuestas de error con HTTP 200.
2. **Early validation:** parar o muestrear y contar **claves únicas**, no
   solo líneas/volumen. Si únicas ≪ líneas, la fuente no avanza — abortar
   y diagnosticar antes de acumular más.
3. **Guarda embebida en el fetcher:** el fetcher debe rastrear claves
   únicas y abortar si una unidad no aporta novedad — nunca acumular
   volumen sin progreso real. `ScrapeHistory` ya deduplica por URL; la
   novedad de *contenido* es la métrica a vigilar.
4. **Mid-run:** al revisar avance, mirar **novedad y distribución**
   (documentos nuevos, dominios cubiertos), no solo volumen y tiempo. "El
   contador sube" no es progreso.
5. **Ante la duda, parar:** si el contenido parece raro, parar, inspeccionar
   el artefacto y reanudar. La resumibilidad existe exactamente para esto.
