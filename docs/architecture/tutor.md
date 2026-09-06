# Tutor architecture

The Tutor is a consumer of materialized knowledge, not an ingestion pipeline.
Its domain includes learning goals, concepts, prerequisites, roadmaps, lessons,
assessment evidence and human approval.

```text
learning goal
  -> diagnosis
  -> roadmap proposal
  -> human approval
  -> learning unit
  -> answer/evidence
  -> assessment
  -> mastery update
  -> next intervention or bounded research request
```

Tutor contracts are authoritative in `contracts/`. Reading a source is exposure,
not mastery. Generated pedagogical fields must retain generation provenance and
source references; approved goals and roadmaps require human approval.

The Tutor provider is replaceable. Qwen/ExLlama operational choices are provider
configuration and benchmark evidence, not Tutor architecture authority.
