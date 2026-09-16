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
```

Rules:

- scraped output belongs under `Landing/web`;
- `Landing/web/scrape_history.db` prevents duplicate downloads;
- databases and derived indexes belong under `outputs/`, not Landing;
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
- **unprocessed / in-flight / unregistered** → stays in Landing;
- `*.db`, hidden files, and `*.pending_delete` are never touched;
- Windows file locks are handled via copy+retry-delete, falling back to a
  `.pending_delete` rename cleaned on the next sweep;
- files already in `Transit/` are re-evaluated every run: promoted to the
  main corpus → `Archive/`; rejected → deleted; still pending → stays.

Runs automatically at the end of `run_full_pipeline` and manually via:

```powershell
.venv\Scripts\python.exe scripts\operations\sweep_landing.py [--dry-run]
```
