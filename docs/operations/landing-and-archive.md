# Landing and Archive

`Landing/` is local intake for authorized artifacts. `Archive/` contains processed
source material. Neither directory is part of the public GitHub surface.

```text
Landing -> registration -> parsing -> chunking -> storage -> indexing -> Archive
```

Rules:

- scraped output belongs under `Landing/web`;
- `Landing/web/scrape_history.db` prevents duplicate downloads;
- databases and derived indexes belong under `outputs/`, not Landing;
- do not delete Landing while it contains unprocessed artifacts;
- do not delete Archive during cleanup;
- use synthetic fixtures under `data/sample/input` for public tests.
