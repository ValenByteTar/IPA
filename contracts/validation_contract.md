# Validation Contract

## Authority

The JSON Schema files in `contracts/*.schema.json` are the **single source of
truth** for the shape of every lab artifact (experiment reports, and future
records). Runtime and test validators MUST validate against the schema file,
not against a hand-maintained duplicate set of rules.

## Levels

```text
structural    field presence, types, enums, patterns
integrity     cross-field and cross-artifact checks (hashes match source,
              artifact_id == content_hash, source files still exist)
```

Both levels are required for a report or manifest to be considered valid.

## Dependency policy

`jsonschema` is a **dev/test-only** dependency. It is installed in the lab
venv for validation and testing but is NOT part of the core runtime
dependencies declared in `requirements.txt` or `pyproject.toml` `[project]
dependencies`. Core adapters must not import it at runtime.

## Required validators

| Artifact | Schema | Validator script |
|---|---|---|
| Experiment report | `contracts/experiment_report.schema.json` | `scripts/validate_experiment_report.py` |
| Processing manifest | `contracts/contract_vocabulary.json` (field vocabulary) | `scripts/validate_contracts.py` |
| Tutor Agent records | `contracts/{learning_goal,concept,roadmap,assessment_result,research_request}.schema.json` | `scripts/validate_tutor_contract.py` |
| Reporter records | `contracts/{reporter_report,reporter_document_decision,topic_link}.schema.json` | `scripts/validate_reporter_contract.py` |

Every validator must:

1. Load the schema file from `contracts/`.
2. Run `jsonschema.validate` (Draft 2020-12) for structural validation.
3. Run additional integrity checks not expressible in JSON Schema.
4. Report all errors, not just the first.
5. Exit non-zero on any error.

## Integrity checks (not expressible in JSON Schema)

For experiment reports:

- `started_at` must be before or equal to `finished_at`.
- `output_hash` must match the hash of the artifact at `output_uri` when
  `output_uri` is present and the file exists.

For processing manifests:

- `artifact_id` must equal `content_hash`.
- `content_hash` must match the SHA-256 of the file at `source_uri` when the
  file exists (integrity mode).
- `byte_size` must match the actual file size (integrity mode).
- `attempts` must be a non-negative integer.
- `status` must belong to the lifecycle states defined in
  `docs/ARCHITECTURE.md`.

For Tutor Agent records:

- source spans must have ordered offsets;
- any effective `generated` or `mixed` field origin requires generation provenance;
- confirmed goals and active/approved roadmap states require an approved human decision;
- roadmap unit order must be unique and contiguous, with unique unit and concept IDs;
- concepts cannot list themselves as prerequisites;
- misconception assessments require misconception evidence;
- research execution requires approval, a bounded budget and an explicit gap;
- completed research requires a job ID and result source references.

## Test discipline

Tests must verify:

- a valid artifact passes;
- each individual constraint violation fails;
- the schema file itself is valid JSON Schema (meta-validation);
- integrity checks catch corruption that structural validation misses.
