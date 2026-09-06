---
name: adr-proposal
description: Prepara propuestas de ADR cuando una decisión cambia una frontera arquitectónica de IPA.
triggers:
  - user
allowed-tools:
  - read
  - grep
  - glob
---

Prepará una propuesta de ADR sin aceptarla automáticamente.

1. Leé las reglas de `AGENTS.md`, `docs/` y los ADRs existentes si los hubiera.
2. Buscá colisiones, supersession y decisiones EKS relacionadas.
3. Verificá que exista evidencia local: tests, benchmark, experiment o postmortem.
4. Diferenciá ADR de Decision, Pattern o Research.
5. Redactá un borrador con estado `Propuesto`, contexto, decisión, consecuencias, alternativas, riesgos y criterios de aceptación.
6. Mantené una única casa para ADRs: `docs/adr/`.
7. No edites ni aceptes un ADR existente; no escribas el borrador hasta que el usuario lo apruebe.

La respuesta debe terminar con:

```text
Estado: Propuesto — requiere aprobación humana.
```
