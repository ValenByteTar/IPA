# Development documentation

Development workflows are intentionally separated from runtime code.

- `AGENTS.md` contains local commands and safety rules.
- `knowledge/` contains EKS records.
- `experiments/` is being canonicalized and archived; new conclusions belong in EKS.
- `scripts/` contains temporary compatibility entrypoints during migration.
- `tools/` contains development-only utilities.

Before deleting or archiving a script, build a reference manifest and verify that
no public entrypoint, dashboard job or test imports it.
