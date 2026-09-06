# Agent Runtime migration plan

IPA owns acquisition and materialization. The Agent Runtime owns bounded query
execution: query interpretation, retrieval, evidence, context, generation and
verification. The Tutor owns learning state and pedagogy.

Migration rules:

- use local contracts and adapters;
- do not import a monolithic external facade;
- preserve legacy behavior behind a compatibility seam;
- introduce one responsibility at a time;
- require budgets, provenance, tests and rollback for every loop;
- promote architectural decisions only after local validation.

The first runtime slice is QueryIR -> EvidenceSet -> ContextPackage. Policy,
Controller, memory and multi-hop remain separate future slices.
