# Dashboard and orchestration architecture

The dashboard is a local control room, not a second domain runtime. Its modules
are separated by responsibility:

```text
server.py
  -> state.py       persistent dashboard/source/config state
  -> jobs.py        process locks, lifecycle and job spawning
  -> api.py         HTTP request handler and endpoint routing
  -> Reporter/IPA services

orchestrator.py
  -> orchestration.py       process coordination
  -> orchestration_ui.py    terminal formatting/progress
```

The dashboard may start jobs, report status and expose local review actions. It
must not own canonical ingestion, indexing or Reporter business rules. Those
belong to `ipa` bounded contexts. Process paths, run IDs, locks, heartbeats and
logs must remain explicit and testable.
