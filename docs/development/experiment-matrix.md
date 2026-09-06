# Experiment matrix

Every experiment records `experiment_id`, `candidate_id`, adapter/tool versions,
hardware, input manifest hash, configuration fingerprint, output hash, timestamps,
status, warnings/errors and resource usage.

Experiments must keep the same corpus, questions and embedding model unless that
is the variable under test. Outputs are isolated and large artifacts remain local.

## Capability tracks

- E0 contract compliance and provenance;
- E1 intake and Landing Zone;
- E2 MIME routing and quarantine;
- E3 PDF parser competition;
- E4 OCR and web acquisition;
- E5 chunking and normalization;
- E6 lexical indexes;
- E7 vector indexes;
- E8 durable queues/workflows;
- E9 semantic enrichment;
- E10 retrieval/response;
- E11 observability.

The authoritative experiment output schema is `contracts/experiment_report.schema.json`.
