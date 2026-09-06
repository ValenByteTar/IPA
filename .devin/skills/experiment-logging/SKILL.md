---
name: experiment-logging
description: Clasifica y registra evidencia local de experimentos, benchmarks, decisiones y postmortems en EKS.
triggers:
  - user
allowed-tools:
  - read
  - grep
  - glob
  - edit
  - write
permissions:
  allow:
    - Read(knowledge/**)
    - Read(outputs/**)
    - Write(knowledge/**)
  deny:
    - Write(src/**)
    - Write(contracts/**)
    - Write(.venv/**)
    - Write(.gitignore)
---

Registrá conocimiento de ingeniería sólo después de una corrida, cambio o incidente con evidencia local.

1. Revisá `knowledge/_schema/metadata.md` y el template de la categoría.
2. Clasificá el resultado primario:
   - Experiment: aprendizaje puntual reproducible;
   - Benchmark: baseline congelada o gate de no-regresión;
   - Decision: decisión local reversible;
   - Postmortem: fallo con causa y prevención;
   - Pattern: solución reusable validada;
   - ADR proposal: cambio de frontera, contrato o principio;
   - Nothing: ruido, evidencia incompleta o resultado no reproducible.
3. Asigná el siguiente ID monotónico de la carpeta destino. Nunca reutilices IDs.
4. Enlazá comandos, configuración, tests y paths de `outputs/` en el documento.
5. No inventes resultados. Si faltan datos, mantené el documento en `draft` o elegí `Nothing`.
6. No edites sustancialmente documentos `accepted`; supersedelos con un nuevo ID.
7. No crees `knowledge/adr/`; los ADRs sólo se proponen en `docs/adr/` y requieren aprobación humana.
8. Ejecutá o indicá el validador EKS antes de finalizar.

La salida debe incluir path, categoría, ID, estado y relaciones creadas.
