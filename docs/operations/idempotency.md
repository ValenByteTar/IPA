# Idempotency and recovery

IPA uses stable artifact, document and chunk identities to make reprocessing
safe. LandingZone identifies artifacts by content hash; DocumentStore uses stable
document/chunk identities; derived indexes must explicitly deduplicate or replace
existing records before publishing.

## Recovery invariants

- Reprocessing the same input must not create duplicate canonical records.
- Derived indexes must be rebuildable from DocumentStore.
- Long-running stages need durable status, retry state and fingerprints.
- Partial vector/enrichment availability must be visible to consumers.
- Original artifacts are immutable; tombstones preserve history.

Use `scripts/validation/` and the experiment reports to verify these invariants.
Do not delete `Landing/` with unprocessed files or `Archive/` as routine cleanup.
