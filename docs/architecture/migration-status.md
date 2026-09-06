# Macro-order migration status

The repository is being migrated from the historical `res023_lab` flat package
to the public `ipa` package and bounded-context layout.

Current phase:

- public surface files and GitHub hygiene created;
- canonical architecture/operations/policy docs created;
- `ipa` bounded-context implementations are active;
- `res023_lab` facade retired (2026-09-06): all callers, tests, scripts and
  dashboard modules import bounded `ipa` paths directly;
- root-level `ipa/*.py` compatibility wrappers removed (2026-09-06);
- remaining flat `scripts/*.py` entrypoints stay until the dashboard and
  orchestrator stop spawning them directly (see deletion-candidates.md);
- full suite remains green.

The migration is intentionally compatibility-first. A bounded context is not
considered moved until imports, tests, CLI behavior and clean-clone smoke pass.
