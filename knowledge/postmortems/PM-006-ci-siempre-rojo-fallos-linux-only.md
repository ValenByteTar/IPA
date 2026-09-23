---
id: PM-006
category: postmortem
status: accepted
created: 2026-09-23
updated: 2026-09-23
author: agent
components: [acquisition, indexes]
tags: [ci, github-actions, linux, windows, portability, subprocess, pathlib]
related: [PAT-007, PM-005]
supersedes: null
superseded_by: null
affects: [".github/workflows/tests.yml", "src/ipa/acquisition/web_scraper.py", "tests/test_index_adapters.py"]
evidence: [".github/workflows/tests.yml", "src/ipa/acquisition/web_scraper.py", "tests/test_index_adapters.py"]
author_model: swe-2
trigger: permit:PW-20260923-08
---

# PM-006 — CI rojo desde el primer commit: 2 fallos Linux-only

## Impacto

Los 9 primeros runs de GitHub Actions (`Primer commit IPA` → `de8870b`)
fallaron todos en `Run tests`. El badge de CI nunca estuvo verde: el
workflow existía pero no protegía nada. Run #10 (`9799bad`) es el primero
en verde: `1144 passed, 2 failed→0, 7 skipped`.

## Causa raíz — dos bugs que solo muerden en POSIX

1. **`tests/test_index_adapters.py:342`** evaluaba
   `subprocess.CREATE_NO_WINDOW | DETACHED_PROCESS` en el assert. Esas
   constantes solo existen en Windows → `AttributeError` en Ubuntu. El
   código de producción (`reranker_adapter.physical_free_vram_mb`) ya
   guardaba con `os.name == "nt"`; el test no.
   Fix: assert condicional — `creationflags == 0` fuera de Windows.
2. **`web_scraper.save_article`** usaba `Path(doc).name` para reducir
   `document_paths` a basenames. En POSIX, `\` no es separador → un path
   `C:\Users\X\run\…-jev-llm-architecture\file.pdf` llegaba íntegro al
   manifest, con los tokens del run dir que el regression test
   prohibía explícitamente.
   Fix: `PureWindowsPath(doc).name` — parsea `\` y `/` como separadores
   en cualquier plataforma.

## Por qué pasaron desapercibidos

Desarrollo y suite completa en Windows; los tests "pasaban" porque el FS
de Windows es case-insensitive y `\` sí separa. La suite en un worktree
limpio local también pasaba (1148) — el único delta real era el OS, no
los datos ni las deps.

## Diagnóstico sin credenciales (técnica reusable)

Los logs de Actions devuelven 403 sin auth y `gh` no estaba autenticado.
Solución: el step de tests captura pytest a `/tmp/pytest.log`, vuelca el
tail a `$GITHUB_STEP_SUMMARY` y emite `::error title=pytest failures::`
con el tail URL-encoded — las anotaciones del check-run **son públicas**
vía HTML de la página del run (sin API). Detalle clave: el warnings
summary de pytest (cientos de líneas de deprecaciones torch/easyocr)
tapaba los `FAILED` → `-p no:warnings` (ningún test usa `pytest.warns`/
`recwarn`) y `awk` para unir líneas con `%0A` (no `paste -d`, que cicla
caracteres como delimitadores).

## Lección

- Tests que asertan comportamiento Windows-only deben guardarse con
  `sys.platform`/`os.name`, no asumir constantes POSIX-ausentes.
- Reducir paths a basename con `PureWindowsPath` (o `ntpath`/`posixpath`
  explícito) cuando el path puede ser de otro OS — `Path().name` es
  dependiente de plataforma del host.
- Hipótesis falsas descartadas en el camino: faltaba `torch` en CI
  (falso — `easyocr`/`sentence-transformers` lo traen), aborto nativo
  (falso — era fallo de aserción en 2 tests), los tests del warnings
  summary como culpables (falso — eran solo emisores de warnings).
