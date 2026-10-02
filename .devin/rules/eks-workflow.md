---
description: "EKS work-permit protocol — acquire scope, close with drafts, evidence honesty"
trigger: always_on
---

# Work permits + EKS discipline (PAT-009)

- Before editing code, acquire a work permit for your scope:
  `.venv/Scripts/python.exe scripts/cli/permit.py acquire --session <id> --scope "<globs>" --task "..."`
  `<id>` is the Devin `session_id` injected by `SessionStart` — a role
  label makes your own permit block your edits. Prefer file-level scopes
  over directory globs when the work allows it (fewer false conflicts).
  An exclusive-scope conflict means STOP and tell the user — never route
  around the block by editing adjacent files instead.
- Types (the STOP lives where the real risk is code): `exclusive` for
  code; `advisory` for docs/knowledge (warns, does not block — the hook
  enforces it as a soft zone); `survey` for read-only exploration.
- The hook renews the heartbeat of YOUR permits on every tool call — a
  batch longer than the TTL no longer expires mid-flight. When the
  session dies, `pid_alive` + TTL reap it automatically (no ghost
  permits).
- On finishing work, close your permits with the session's harvest:
  `permit.py close --permit PW-... --notes "..." --eks-draft <ID>` —
  invoke skill `session-closeout` for the full protocol. A permit closed
  by `SessionEnd` without `--eks-draft` leaves an `unharvested` marker that
  the next `SessionStart` reports; `Stop` blocks once per session asking
  for this closeout. Closing without harvesting is not a neutral act.
- Evidence honesty: `status: accepted` requires `evidence:` paths that
  exist on disk (or an existing cited artifact). No evidence → `draft`.
- New records get `affects` when they govern concrete paths — without it
  they never enter the `eks_governing`/permit injection circuit.
- Records produced under a permit declare `trigger: permit:PW-*`.
