---
description: "EKS work-permit protocol — acquire scope, close with drafts, evidence honesty"
trigger: always_on
---

# Work permits + EKS discipline (PAT-009)

- Before editing code, acquire a work permit for your scope:
  `.venv/Scripts/python.exe scripts/cli/permit.py acquire --session <id> --scope "<globs>" --task "..."`
  An exclusive-scope conflict means STOP and tell the user — never route
  around the block by editing adjacent files instead.
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
