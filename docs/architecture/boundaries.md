# Ownership and boundaries

## Materialization plane

| Responsibility | Owner |
|---|---|
| Artifact registration | Landing Zone |
| MIME/safety | MIME router and safety adapters |
| Canonical parsing | parser adapters |
| Chunk identity and spans | chunker/contracts |
| Canonical documents/chunks | DocumentStore |
| Lexical/vector views | index adapters |
| Generated enrichment | enrichment subsystem |
| Trace events | observability subsystem |

## Consumer plane

| Responsibility | Owner |
|---|---|
| Query interpretation | planner |
| Retrieval | retrieval adapter |
| Ranking/selection | reranker/evidence selector |
| Context | context builder |
| Generation | model provider/generation capability |
| Verification | claims/evaluation |
| Next action | future bounded policy/orchestrator |

## Product capabilities

Reporter and Tutor consume materialized knowledge through explicit contracts. They
must not silently become alternate ingest pipelines or mutate canonical source
text. EKS is dev-time only and is exposed through its own read-only MCP server.

The package migration target is `ipa`; `res023_lab` remains a temporary compatibility
facade while callers migrate.
