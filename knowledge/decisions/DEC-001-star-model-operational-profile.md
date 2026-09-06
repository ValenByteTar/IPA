---
id: DEC-001
category: decision
status: proposed
created: 2026-09-05
updated: 2026-09-05
author: human
components: [tutor, providers, configuration, performance]
tags: [qwen3.5-9b, exllama, exl3, mtp, no-think, batch, provisional]
related: [RES-004, EXP-004, BM-006]
supersedes: null
superseded_by: null
---

# DEC-001 — Perfil operacional provisional del modelo estrella

## Contexto

IPA ya integra `ExL3Provider` y documenta Qwen3.5-9B EXL3 3.0bpw + MTP como modelo estrella. La evidencia resumida favorece esta combinación para el hardware local actual, pero la recomendación completa todavía necesita reconstrucción de provenance y pruebas de estabilidad operacional.

## Decisión propuesta

Usar provisionalmente el siguiente perfil cuando se ejecute el Tutor Agent o inferencia paralela de alto volumen:

```text
model: Qwen3.5-9B EXL3 3.0bpw
engine: ExLlamaV3
use_mtp: true
batch_size: 6 en procesamiento paralelo, 1 en interacción
context_length: 4096
mtp_cache_tokens: 4096
temperature: 0.0
no_think: true
```

No activar deliberación automáticamente mientras el benchmark disponible indique daño neto o no exista evidencia de utilidad positiva en el flujo real.

## Consecuencias

- Se aprovecha la implementación existente del provider.
- El perfil cabe con margen reportado en la GPU de aproximadamente 6 GB.
- Se acepta una debilidad conocida en diagnóstico pedagógico.
- El uso de MTP introduce variabilidad residual aun con temperatura 0.
- Cambiar modelo, engine, cuantización o hardware requiere actualizar benchmark y revisar esta decisión.

## Condiciones de aceptación

Esta decisión no pasa a `accepted` hasta que `BM-006` tenga:

- artefactos crudos y run IDs verificables;
- entorno y dependencias reconciliados con IPA;
- prueba de estabilidad prolongada;
- evaluación de diagnóstico pedagógico;
- comparación limpia contra los controles definidos;
- revisión de fallos y recuperación.

## Rollback

Volver al perfil anterior o a un provider alternativo mediante configuración/composition root, sin modificar contratos del corpus. El proveedor es una implementación reemplazable y no una autoridad de arquitectura.
