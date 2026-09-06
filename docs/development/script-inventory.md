# Script inventory during migration

Scripts are being grouped by responsibility. The existing flat files remain in
place until their references are migrated and the equivalent smoke test passes.

## Public/operations entrypoints

- `run_fast_path.py`
- `run_continuous_pipeline.py`
- `run_web_scrape.py`
- `run_reporter.py`
- `web_dashboard.py`
- `orchestrator.py`
- `dashboard_watchdog.py`
- `run_trace_query.py`
- `test_exl3_provider.py`

Target: `scripts/cli/` or `scripts/operations/`, implemented as thin wrappers
around `ipa` services.

## Validation

- `validate_contracts.py`
- `validate_experiment_report.py`
- `validate_reporter_contract.py`
- `validate_tutor_contract.py`
- `validate_eks.py`

Target: `scripts/validation/`.

## Benchmarks and reproducible research

- `run_index_benchmark.py`
- `run_parser_benchmark.py`
- `run_retrieval_eval.py`
- `run_enrichment_eval.py`
- `run_enrichment_experiment.py`
- `run_enrichment_batched.py`
- `run_adaptive_rechunk.py`
- `compare_adaptive_retrieval.py`
- `evaluate_reporter.py`

Target: `scripts/benchmarks/`. Each runner must write a small manifest and keep
large outputs outside the public repository.

## Archive candidates

- `_*.py` scratch scripts;
- `proc_*.py` process wrappers after orchestrator dependency audit;
- one-off diagnostics and repair scripts;
- generated helper code under `outputs/`.

These are copied/moved to `local_archive/` only after a reference manifest is
written. They are not deleted during the first ordering pass.
