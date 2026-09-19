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

## Runtime (implementation)

- State: `outputs/agent/tutor.db` (`TutorStore`) — `user_topic_records`,
  append-only `user_evidence`, `roadmaps`, `research_requests`,
  `unit_progress`, `unit_summaries`, `tutor_focus`, `session_roadmap`.
- Runtime: `src/ipa/tutor/tutor_runtime.py` (`TutorSession`, `TutorStore`,
  `DiagnosisResult`); dashboard wiring: `src/ipa/tutor/tutor_chat.py`
  (`TutorChatDriver`, per-`session_id` state machine, `role="tutor"` on
  `/api/agent/chat/stream`).

### Roadmap gate and debate

`propose_roadmap` → `approve`/`reject` (human) → `activate` (approved only).
With a proposed roadmap, any message that is not approve/reject is treated as
feedback: the LLM re-proposes (v+1, `previous_roadmap_id`) and the previous
version becomes `superseded` (`supersede_roadmap` requires `HumanApproval` —
the debate *is* the human action). The gate still applies to the revision.

**Absorbing scaffold** (`_shape_units`): imperfect LLM proposals never
dead-end. Unknown concept_ids are dropped, repeats deduped, the count is
topped up to the contract minimum (3) with unused concepts and truncated to
the maximum (7). Residual failures reach the learner as a friendly message
(the exception goes to the log). When no retrieved hit mentions the requested
topic, the reply carries a transparency note.

### Roadmap focus (cross-session)

Clicking a stepper card, or activating a roadmap, calls
`POST /api/tutor/roadmap/focus`: the current session adopts that roadmap and
the focus persists (`tutor_focus`, single row). New or idle sessions adopt it
on the next message without a new topic (`_adopt_focus`); rejecting clears the
focus if it pointed there. Chat indicator: chip "📍 topic · unit N/M"
(`GET /api/tutor/focus`, refreshed on session/done/decisions).

Session summaries carry a deterministic roadmap tag —
`[roadmap:<id> · tema: X · unidad N/M · estado]` — built from
`session_roadmap` (written by the driver), so the tagged summary is
cross-session retrievable through `recall_memory`.

### Unit progress

`unit_progress` (`roadmap_id`, `unit_order`, `status`) is additive — the
Roadmap contract stays immutable. Statuses: `pending|current|done`; unit 1 is
seeded `current` on activation. Deterministic advance in lessons ("siguiente
unidad", "ya entendí", "avancemos" → `_ADVANCE_RE`).
`GET /api/tutor/roadmaps` returns units + status + titles (source domain,
never doc_ids) + topic mastery; the stepper renders it
(`web/static/app.js renderTutorRoadmaps`, CSS `.rm-*`).

### Lesson length

The base identity (`configs/agent_identity.yaml`) scopes "1-5 oraciones" to the
general chat and gives the tutor role an explicit exception (rich
explanations: definition + example + connection); `TUTOR_POLICY` adds
"explicaciones ricas (2-4 párrafos)" and the lesson budget is 1536
`max_new_tokens`. Measured live: ~90 → 184 words per lesson.

### Research and gates in the chat

Insufficient corpus (<3 concepts) → `ResearchRequest` (human gate) → background
executor; on completion an episode is written into the user's session and a
topic-less message ("dale") resumes the stored topic. Tutor research budget:
15 sources / 300 s (general chat `research_topic`: default 5, cap 20). While an
approved research is in flight the driver answers deterministically ("aguardamos
a que llegue la información de la fuente web") — no LLM, no proposed steps.
Frontend gates (Aprobar/Rechazar/Debatir) are re-mounted from
`/api/agent/approvals` after each canonical re-render (`mountPendingTutorGates`).

