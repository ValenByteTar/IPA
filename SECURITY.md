# Security Policy

IPA processes local documents, web content, generated artifacts and optional local
models. Do not commit credentials, tokens, private corpus data, model weights,
local databases, or generated indexes.

## Reporting

Report security issues privately to the repository owner rather than publishing
sensitive details in an issue.

## Security boundaries

- Scraping uses explicit domain allowlists and bounded requests.
- MCP file ingestion is restricted to controlled inbox paths.
- Localhost, private and reserved network targets must be blocked by fetch/MCP
  boundaries.
- Canonical source text remains distinguishable from generated enrichment.
- Generated claims require provenance and validation before promotion.
- Outputs and local corpora are not part of the public GitHub surface.
