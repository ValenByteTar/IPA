# Landing, Transit and Archive

`Landing/` is local intake for authorized artifacts — a transit zone, not
storage. `Transit/` holds processed artifacts awaiting confirmation for the
main corpus. `Archive/` contains processed source material already admitted to
the main corpus. None of these directories is part of the public GitHub surface.

```text
Landing -> registration -> parsing -> chunking -> storage -> indexing
    |
    +-- approved (in main corpus) --------------> Archive/
    +-- pending confirmation (staging/review) --> Transit/ --> Archive/ on promote
    |                                                        -> delete on reject
    +-- rejected (failed / human-rejected) ------> deleted
    +-- no_text (parsed OK, zero usable text) ---> deleted
```

Rules:

- scraped output belongs under `Landing/web`;
- `Landing/web/scrape_history.db` prevents duplicate downloads;
- databases and derived indexes belong under `outputs/`, not Landing;
- **agent research does not use `Landing/web` as its work dir**: each
  `research_topic` run scrapes into `outputs/agent/research/<run_id>/` and
  ingests only that dir, so it never inherits a concurrent pipeline's landing
  contents nor races it for the same tree (PM-004). Its material reaches the
  canonical corpus through ingestion + `provenance=agent_research` (PAT-003),
  and the run dir is kept as the source-material audit trail (retention is
  manual — no auto-pruning);
- **the rejected-doc review worker** (idle 60 s and the T2 batched pass)
  promotes into `outputs/agent/research/review/` for the same reason — it used
  to ingest the shared `Landing/web`;
- nothing processed stays in Landing — every processed artifact leaves for
  Archive, Transit, or deletion;
- do not delete Landing while it contains unprocessed artifacts;
- do not delete Archive or Transit during cleanup;
- use synthetic fixtures under `data/sample/input` for public tests.

## Transit-zone enforcement (landing sweep)

The sweep (`src/ipa/ingestion/landing_sweep.py`) classifies every registered
artifact using the corpus `landing.db` registries, `document_store.db`
membership, and `outputs/agent/topic_clusters.db` (curation `review_status`
and `promotion_queue`):

- **approved** — a live document exists in the MAIN corpus
  (`outputs/experiments/E12-corpus`) → moved to `Archive/`, preserving the
  Landing-relative path;
- **pending confirmation** — ingested into a staging corpus but not in main,
  queued in `promotion_queue`, or a curation decision is still
  `review_status=pending`/`changes_requested` → moved to `Transit/`;
- **rejected** — `failed` in every registry and in no store, or a human set
  `review_status=rejected` → deleted;
- **no_text** — the parse succeeded but produced zero chunks (image-only PDF
  whose OCR returned nothing, empty/whitespace file). The fast path stores
  **no document** for these — an empty doc would only be rejected by curation
  later — and marks the artifact `no_text`, so the sweep deletes the file
  (Landing and the Transit re-evaluation alike) instead of it lingering as
  "indexed" forever;
- **unprocessed / in-flight / unregistered** → stays in Landing;
- `*.db`, hidden files, and `*.pending_delete` are never touched;
- Windows file locks are handled via copy+retry-delete, falling back to a
  `.pending_delete` rename cleaned on the next sweep;
- files already in `Transit/` are re-evaluated every run: promoted to the
  main corpus → `Archive/`; rejected → deleted; still pending → stays.

Promotion is vector-gated (PM-004): `promote_documents_to_main` only purges
the staging copy once **every live chunk of the batch exists in main LanceDB**.
If coverage is incomplete — typically while a drain is still backfilling — the
promotion is **deferred**: the source document, its staging chunks/vectors and
its `promotion_queue` entry stay untouched, and the next cycle retries
idempotently. An unreadable main LanceDB also defers rather than risking a
purge without vectors. Emergency opt-out: `IPA_PROMOTION_REQUIRE_VECTORS=0`.

The source purge itself is also retry-gated: each step (DocumentStore
tombstone, BM25 FTS drop, LanceDB delete) must succeed or the batch defers
and the queue retries. The BM25 step tolerates a `bm25_index.db` briefly
locked by a concurrent fast-path ingestion (bounded retries with
`BEGIN IMMEDIATE` + busy timeout); a definitive failure is reported as
`incomplete_steps` and never silently logged — a partial purge used to
leave live FTS rows for documents already gone from the staging store.

Runs automatically at the end of `run_full_pipeline` and manually via:

```powershell
.venv\Scripts\python.exe scripts\operations\sweep_landing.py [--dry-run]
```
