---
name: refactoring
description: Refactorizar preservando comportamiento — sin estética, con tests como prueba, y PAT/DEC cuando aparece un patrón o una frontera.
triggers:
  - user
  - model
allowed-tools:
  - read
  - grep
  - glob
  - exec
  - edit
  - write
permissions:
  allow:
    - Read(**)
  deny:
    - Write(contracts/**)
    - Write(.gitignore)
    - Write(.venv/**)
---

1. **Nunca refactorices por estética.** Hacelo solo si mejora arquitectura,
   legibilidad, extensibilidad o testabilidad — y decí cuál de las cuatro.
2. **Leé lo que gobierna los paths** antes de moverlos:
   `eks_governing(paths)` (incluye rejected/superseded). Si el refactor toca
   una frontera, necesita una DEC (skill `adr-proposal`), no una decisión
   implícita en el diff.
3. **Comportamiento preservado = evidencia, no intención.** Corré los tests
   del área antes y después. Si no hay tests que cubran lo que movés,
   escribilos primero o declaralo explícitamente como no cubierto.
4. **Interfaces públicas**: no las cambies sin aprobación. `contracts/` es
   autoridad; los tools se integran por adapters, no por conveniencia.
5. **No mezcles refactor con feature.** Si el diff hace las dos cosas,
   separalo en dos cambios.
6. **Si aparece un patrón reusable**, proponé un `PAT-*`
   (`scripts/cli/eks_new.py pattern --title "..."`). Si cambia una frontera,
   proponé una `DEC-*` (`adr-proposal`).

Salida: qué se movió y por qué (cuál de las cuatro mejoras), evidencia de que
el comportamiento no cambió (comando + resultado), y records propuestos.
