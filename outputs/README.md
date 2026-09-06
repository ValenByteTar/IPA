# Outputs

This directory holds all lab-produced artifacts. **Nothing here is a source of
truth for production.** Outputs are evidence, not changes.

## Structure

```
outputs/
  manifests/             Landing manifests (JSONL)
  experiments/
    E0/                  Contract compliance
    E1/                  Intake and Landing Zone
    E2/                  MIME router
    E3/                  PDF parser competition
    E4/                  OCR
    E5/                  Chunking and normalization
    E6/                  Lexical index
    E7/                  Vector index
    E8/                  Queue/workflow
    E9/                  Semantic enrichment
    E10/                 End-to-end retrieval/response
    E11/                 Observability
  logs/                  Run logs and traces
```

## Rules

- Every experiment writes only to its own `experiments/<E#>/` namespace.
- Each run should include: input manifest hash, configuration fingerprint,
  output artifacts, and a validated `report.json` (see
  `contracts/experiment_report.schema.json`).
- Never overwrite a previous run's artifacts; use run-indexed subdirectories.
- Never point any output at production indexes or databases.
