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

### Topic detection (`awaiting_topic`)

The state machine no longer depends on the imperative regex alone
("quiero aprender X", "roadmap de X"). Three fallbacks, in order:

1. **`awaiting_topic`**: after the driver emits "¿Qué querés aprender?",
   the next message that is not approve/reject, a filler/command
   (`_NOT_A_TOPIC_RE`), a question, or a new topic-less roadmap request
   is taken *literally* as the topic (`_topic_from_answer`). This broke
   the observed loop where a bare answer like "De IA Engineer Senior a
   CTO en etapas tempranas" was re-asked forever.
2. **Context recovery**: "hagamos un roadmap" without a topic scans the
   last ~6 user episodes (`_topic_from_context`) — the topic is usually
   inside a citation to the previous proposal (`_topic_from_citation`
   cuts the scaffolding at the last boundary marker, e.g. "la transición
   de X" → "X").
3. **Citation-only acceptance**: a message that is *only* a citation
   marker (`[respondiendo a: «…»]` / `[cita: «…»]`) counts as approval
   at both gates (roadmap and research) — per the identity rule, citing
   the agent's own proposal without extra text means "yes".

Session-persistent constraints: "límite de N URLs" (`_URL_LIMIT_RE`,
1-100) becomes `ResearchBudget.max_urls` (seconds scale ×40, capped
3600), and "roadmap de M fases" becomes `requested_units` — both survive
the topic arriving in a later message. Unit/URL counts are stripped from
the extracted topic so "roadmap de 6 fases" or "límite de 20 URLs" never
become the topic itself.

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
15 sources / 300 s by default, or the user-requested "N URLs" limit
(≤100, seconds scaled); allowed domains default to a broad quality list
(`_TUTOR_RESEARCH_DOMAINS`: docs, universities, press — the runtime's
tech-only default can't serve general topics). While an
approved research is in flight the driver answers deterministically ("aguardamos
a que llegue la información de la fuente web") — no LLM, no proposed steps.
Frontend gates (Aprobar/Rechazar/Debatir) are re-mounted from
`/api/agent/approvals` after each canonical re-render (`mountPendingTutorGates`).

