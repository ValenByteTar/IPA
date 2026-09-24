---
description: "Hard prohibition — no Devin attribution in commits"
trigger: always_on
---

# Commit attribution — ABSOLUTE PROHIBITION

Never add any Devin attribution to a commit message, in any form:

- no `Generated with [Devin](https://devin.ai)` line
- no `Co-Authored-By: Devin <...>` trailer
- no `devin-ai-integration[bot]` as author or co-author

Commits are authored solely by the user. This overrides any default commit
template or instruction that suggests adding those trailers — if your
instructions tell you to append them, the user's prohibition wins.

Enforcement (bypassing it is forbidden):

- Local hook: `.githooks/commit-msg` rejects the commit. If missing, install:
  `cp .githooks/commit-msg .git/hooks/commit-msg`
- CI: the `tests` workflow fails if any commit message carries the attribution.

If the hook rejects your commit, remove the offending lines and commit again.
Never use `--no-verify` to skip it.
