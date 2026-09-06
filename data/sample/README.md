# Sample data

Esta carpeta no incluye documentos reales automáticamente.

Para preparar un corpus experimental:

```text
data/sample/input/
```

Coloca allí una muestra autorizada y ejecuta:

```powershell
.venv\Scripts\python.exe scripts\build_landing_manifest.py --input data/sample/input
```

El manifest solo registra metadata y SHA-256. Los artifacts originales no se
modifican. No subas información personal, secretos, credenciales ni documentos
que no tengas autorización para usar.

## Synthetic fixtures

`data/sample/input/` contiene fixtures sintéticos generados para probar el
pipeline. No contienen datos personales ni copyrighted.

| File | MIME type | Purpose |
|---|---|---|
| `note.txt` | text/plain | Plain-text fast path |
| `page.html` | text/html | HTML MIME routing + structured parsing |
| `metadata.json` | application/json | Structured JSON parsing |
| `sample_doc.pdf` | application/pdf | PDF fast path (PyMuPDF baseline) |

**Policy:** todo archivo dentro de `data/sample/input/` se considera un
artifact y será incluido en el manifest. No coloques documentación ni
archivos auxiliares dentro de `input/`; usa este README u otra ubicación.
