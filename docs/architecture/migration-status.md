# Macro-order migration status

The repository is being migrated from the historical `res023_lab` flat package
to the public `ipa` package and bounded-context layout.

Current phase:

- public surface files and GitHub hygiene created;
- canonical architecture/operations/policy docs created;
- `ipa` bounded-context implementations are active;
- `res023_lab` is now a compatibility facade backed by `ipa`;
- remaining flat compatibility modules are scheduled for retirement after a review window;
- full suite remains green.

The migration is intentionally compatibility-first. A bounded context is not
considered moved until imports, tests, CLI behavior and clean-clone smoke pass.
