"""Tests for the cognitive layer: task planner, strategic memory, skill library,
uncertainty tracking, and transversal user model (puntos 2-6, 8).

Each module is tested in isolation with a temp SQLite store. No network,
no LLM, no corpus required — all deterministic.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Set test stores BEFORE importing the modules so they use temp paths.
_TMP = Path(tempfile.mkdtemp(prefix="ipa_cognitive_"))
os.environ["IPA_TASK_STORE"] = str(_TMP / "task_store.db")
os.environ["IPA_STRATEGIC_STORE"] = str(_TMP / "strategic.db")
os.environ["IPA_SKILL_STORE"] = str(_TMP / "skill_library.db")
os.environ["IPA_UNCERTAINTY_STORE"] = str(_TMP / "uncertainty.db")
os.environ["IPA_USER_MODEL_STORE"] = str(_TMP / "user_model.db")

from ipa.agent.task_planner import (
    Task, SubTask, TaskStore, Planner, TaskExecutor,
    DEFAULT_MAX_SUBTASKS, DEFAULT_MAX_TOOLS_PER_SUBTASK,
)
from ipa.agent.strategic_memory import (
    Principle, StrategicMemoryStore, StrategicReflector, render_active_principles,
)
from ipa.agent.skill_library import (
    Skill, SkillLibraryStore, SkillDetector, render_active_skills,
    MIN_OCCURRENCES, MIN_TOOLS_IN_SKILL,
)
from ipa.agent.uncertainty import (
    TopicConfidence, ResearchProposal, UncertaintyStore,
    UncertaintyTracker, ActiveResearchAgenda,
    CONFIDENCE_THRESHOLD, MIN_OBSERVATIONS,
    render_uncertainty_context,
)
from ipa.agent.user_model import (
    UserGoal, UserInterest, UserPreference, UserModelStore, UserModelInferer,
    render_user_model_context,
)


# ---------------------------------------------------------------------------
# TaskStore + Planner + TaskExecutor (puntos 2+3)
# ---------------------------------------------------------------------------

class TestTaskStore:
    def test_save_and_get_task(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        subtasks = [
            SubTask(id=0, action="search_corpus", args={"query": "test"}, why="buscar"),
            SubTask(id=1, action="compile_report", args={"query": "test"}, why="reportar"),
        ]
        task = Task(
            task_id="task:test1", goal="test goal", subtasks=subtasks,
            budget={"max_subtasks": 6, "max_tools_per_subtask": 3},
            status="planned", current_subtask=0,
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        )
        store.save_task(task)
        loaded = store.get_task("task:test1")
        assert loaded is not None
        assert loaded.goal == "test goal"
        assert len(loaded.subtasks) == 2
        assert loaded.subtasks[0].action == "search_corpus"
        assert loaded.progress == 0.0
        store.close()

    def test_update_subtask_status_completed(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        subtasks = [
            SubTask(id=0, action="search_corpus", args={"query": "x"}, why="y"),
            SubTask(id=1, action="compile_report", args={"query": "x"}, why="y"),
        ]
        task = Task(
            task_id="task:t1", goal="g", subtasks=subtasks,
            budget={"max_subtasks": 6, "max_tools_per_subtask": 3},
            status="planned", current_subtask=0,
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        )
        store.save_task(task)
        store.update_subtask_status("task:t1", 0, "completed", result_summary="ok")
        loaded = store.get_task("task:t1")
        assert loaded.subtasks[0].status == "completed"
        assert loaded.current_subtask == 1
        assert loaded.status == "planned"  # still has pending subtasks
        store.close()

    def test_update_subtask_status_last_completes_task(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        subtasks = [
            SubTask(id=0, action="search_corpus", args={"query": "x"}, why="y"),
        ]
        task = Task(
            task_id="task:t2", goal="g", subtasks=subtasks,
            budget={"max_subtasks": 6, "max_tools_per_subtask": 3},
            status="running", current_subtask=0,
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        )
        store.save_task(task)
        store.update_subtask_status("task:t2", 0, "completed", result_summary="ok")
        loaded = store.get_task("task:t2")
        assert loaded.status == "completed"
        store.close()

    def test_list_tasks_by_status(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        for i, status in enumerate(["planned", "running", "completed"]):
            task = Task(
                task_id=f"task:l{i}", goal=f"g{i}", subtasks=[],
                budget={"max_subtasks": 6, "max_tools_per_subtask": 3},
                status=status, current_subtask=0,
                created_at=f"2026-01-0{i+1}T00:00:00Z", updated_at=f"2026-01-0{i+1}T00:00:00Z",
            )
            store.save_task(task)
        running = store.list_tasks(status="running")
        assert len(running) == 1
        assert running[0].status == "running"
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 3
        store.close()

    def test_get_active_task(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        # No active tasks initially
        assert store.get_active_task() is None
        # Add a planned task
        task = Task(
            task_id="task:active1", goal="g", subtasks=[],
            budget={"max_subtasks": 6, "max_tools_per_subtask": 3},
            status="planned", current_subtask=0,
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        )
        store.save_task(task)
        active = store.get_active_task()
        assert active is not None
        assert active.task_id == "task:active1"
        store.close()


class TestPlanner:
    def test_plan_deterministic_investigar(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=None)  # no LLM → deterministic
        task = planner.plan("investigá fotónica")
        assert task.goal == "investigá fotónica"
        assert len(task.subtasks) >= 2
        assert task.subtasks[0].action == "search_corpus"
        # "investigar" template includes research_topic + compile_report
        actions = [s.action for s in task.subtasks]
        assert "research_topic" in actions
        assert "compile_report" in actions
        assert task.status == "planned"
        store.close()

    def test_plan_deterministic_reporte(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=None)
        task = planner.plan("hacé un reporte sobre quantum computing")
        actions = [s.action for s in task.subtasks]
        assert "search_corpus" in actions
        assert "compile_report" in actions
        store.close()

    def test_plan_deterministic_buscar(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=None)
        task = planner.plan("buscá RAG")
        assert len(task.subtasks) == 1
        assert task.subtasks[0].action == "search_corpus"
        store.close()

    def test_plan_budget_respected(self, tmp_path):
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=None)
        task = planner.plan("investigá X", max_subtasks=2)
        assert len(task.subtasks) <= 2
        store.close()

    def test_plan_with_invalid_llm_falls_back(self, tmp_path):
        """If LLM produces invalid JSON, fall back to deterministic."""
        class FakeProvider:
            def generate_chat(self, messages, max_new_tokens=600):
                class R:
                    text = "not json at all"
                    error = None
                return R()
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=FakeProvider())
        task = planner.plan("investigá fotónica")
        # Should fall back to deterministic
        assert len(task.subtasks) >= 1
        assert task.subtasks[0].action == "search_corpus"
        store.close()

    def test_plan_with_valid_llm_json(self, tmp_path):
        """If LLM produces valid JSON with valid tools, use it."""
        class FakeProvider:
            def generate_chat(self, messages, max_new_tokens=600):
                class R:
                    text = json.dumps({
                        "goal": "test",
                        "subtasks": [
                            {"action": "search_corpus", "args": {"query": "test"}, "why": "buscar"},
                            {"action": "compile_report", "args": {"query": "test"}, "why": "reportar"},
                        ],
                        "budget": {"max_subtasks": 6, "max_tools_per_subtask": 3},
                    })
                    error = None
                return R()
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=FakeProvider())
        task = planner.plan("test")
        assert len(task.subtasks) == 2
        assert task.subtasks[0].action == "search_corpus"
        assert task.subtasks[1].action == "compile_report"
        store.close()

    def test_plan_with_invalid_tool_in_llm_json_filters(self, tmp_path):
        """LLM proposes a non-existent tool → that subtask is dropped."""
        class FakeProvider:
            def generate_chat(self, messages, max_new_tokens=600):
                class R:
                    text = json.dumps({
                        "goal": "test",
                        "subtasks": [
                            {"action": "search_corpus", "args": {"query": "test"}, "why": "ok"},
                            {"action": "nonexistent_tool", "args": {}, "why": "bad"},
                        ],
                        "budget": {"max_subtasks": 6, "max_tools_per_subtask": 3},
                    })
                    error = None
                return R()
        store = TaskStore(tmp_path / "task.db")
        planner = Planner(store, provider=FakeProvider())
        task = planner.plan("test")
        # Only the valid subtask should remain
        assert len(task.subtasks) == 1
        assert task.subtasks[0].action == "search_corpus"
        store.close()


# ---------------------------------------------------------------------------
# Strategic memory (punto 4)
# ---------------------------------------------------------------------------

class TestStrategicMemory:
    def test_save_and_get_principle(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        p = Principle(
            principle_id="principle:test1", kind="tool_pattern",
            pattern="search → compile", principle="test principle",
            evidence={"count": 5}, confidence=0.8,
            status="pending", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_principle(p)
        loaded = store.get_principle("principle:test1")
        assert loaded is not None
        assert loaded.principle == "test principle"
        assert loaded.confidence == 0.8
        store.close()

    def test_decide_principle_approved(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        p = Principle(
            principle_id="principle:d1", kind="tool_pattern",
            pattern="p", principle="p", evidence={}, confidence=0.5,
            status="pending", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_principle(p)
        store.decide("principle:d1", approved=True)
        loaded = store.get_principle("principle:d1")
        assert loaded.status == "approved"
        assert loaded.decided_by == "human"
        store.close()

    def test_active_principles_only_approved(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        for i, status in enumerate(["pending", "approved", "rejected"]):
            p = Principle(
                principle_id=f"principle:a{i}", kind="tool_pattern",
                pattern=f"p{i}", principle=f"p{i}", evidence={},
                confidence=0.5, status=status, proposed_at="2026-01-01T00:00:00Z",
            )
            store.save_principle(p)
        active = store.active_principles()
        assert len(active) == 1
        assert active[0].status == "approved"
        store.close()

    def test_render_active_principles_empty(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        assert render_active_principles(store) == ""
        store.close()

    def test_render_active_principles_content(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        p = Principle(
            principle_id="principle:r1", kind="tool_pattern",
            pattern="p", principle="test principle text", evidence={},
            confidence=0.8, status="approved", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_principle(p)
        rendered = render_active_principles(store)
        assert "test principle text" in rendered
        store.close()

    def test_reflector_detects_tool_patterns(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        reflector = StrategicReflector(store)
        # 5 episodes with the same tool sequence
        episodes = []
        for i in range(5):
            episodes.append({
                "session_id": f"sess{i}", "turn_role": "assistant",
                "content": f"result {i}", "tool_calls": ["search_corpus", "compile_report"],
            })
        proposals = reflector.reflect(episodes)
        # Should detect the search_corpus → compile_report pattern
        assert len(proposals) >= 1
        tool_pat = [p for p in proposals if p.kind == "tool_pattern"]
        assert len(tool_pat) >= 1
        store.close()

    def test_reflector_dedupes_existing_patterns(self, tmp_path):
        store = StrategicMemoryStore(tmp_path / "strat.db")
        reflector = StrategicReflector(store)
        episodes = [
            {"session_id": "s1", "turn_role": "assistant", "content": "r",
             "tool_calls": ["search_corpus", "compile_report"]}
            for _ in range(5)
        ]
        # First reflection creates proposals
        first = reflector.reflect(episodes)
        assert len(first) >= 1
        # Second reflection with same episodes should not create duplicates
        second = reflector.reflect(episodes)
        assert len(second) == 0
        store.close()


# ---------------------------------------------------------------------------
# Skill library (punto 5)
# ---------------------------------------------------------------------------

class TestSkillLibrary:
    def test_save_and_get_skill(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        s = Skill(
            skill_id="skill:test1", name="research_and_report",
            description="test skill", steps=[{"action": "search_corpus"}, {"action": "compile_report"}],
            trigger="when user wants report", occurrences=5, confidence=0.5,
            status="pending", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_skill(s)
        loaded = store.get_skill("skill:test1")
        assert loaded is not None
        assert loaded.name == "research_and_report"
        assert len(loaded.steps) == 2
        store.close()

    def test_decide_skill_approved(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        s = Skill(
            skill_id="skill:d1", name="n", description="d", steps=[],
            trigger="t", occurrences=3, confidence=0.5,
            status="pending", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_skill(s)
        store.decide("skill:d1", approved=True)
        loaded = store.get_skill("skill:d1")
        assert loaded.status == "approved"
        store.close()

    def test_active_skills_only_approved(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        for i, status in enumerate(["pending", "approved"]):
            s = Skill(
                skill_id=f"skill:a{i}", name=f"n{i}", description="d", steps=[],
                trigger="t", occurrences=3, confidence=0.5,
                status=status, proposed_at="2026-01-01T00:00:00Z",
            )
            store.save_skill(s)
        active = store.active_skills()
        assert len(active) == 1
        store.close()

    def test_render_active_skills_empty(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        assert render_active_skills(store) == ""
        store.close()

    def test_render_active_skills_content(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        s = Skill(
            skill_id="skill:r1", name="my_skill", description="does X", steps=[],
            trigger="when X", occurrences=5, confidence=0.8,
            status="approved", proposed_at="2026-01-01T00:00:00Z",
        )
        store.save_skill(s)
        rendered = render_active_skills(store)
        assert "my_skill" in rendered
        assert "does X" in rendered
        store.close()

    def test_detector_proposes_skill_for_repeated_pattern(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        detector = SkillDetector(store)
        # 4 sessions with the same 2-tool sequence
        episodes = []
        for i in range(4):
            episodes.append({
                "session_id": f"sess{i}", "tool_calls": ["search_corpus", "compile_report"],
            })
        proposals = detector.detect(episodes)
        assert len(proposals) >= 1
        s = proposals[0]
        assert len(s.steps) == 2
        assert s.occurrences >= MIN_OCCURRENCES
        assert s.status == "pending"
        store.close()

    def test_detector_ignores_single_tool_sequences(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        detector = SkillDetector(store)
        episodes = [
            {"session_id": f"s{i}", "tool_calls": ["search_corpus"]}
            for i in range(5)
        ]
        proposals = detector.detect(episodes)
        # Single-tool sequences don't meet MIN_TOOLS_IN_SKILL
        assert len(proposals) == 0
        store.close()

    def test_detector_dedupes_existing(self, tmp_path):
        store = SkillLibraryStore(tmp_path / "skill.db")
        detector = SkillDetector(store)
        episodes = [
            {"session_id": f"s{i}", "tool_calls": ["search_corpus", "compile_report"]}
            for i in range(5)
        ]
        first = detector.detect(episodes)
        assert len(first) >= 1
        second = detector.detect(episodes)
        assert len(second) == 0
        store.close()


# ---------------------------------------------------------------------------
# Uncertainty + active research agenda (punto 6)
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_record_observation_creates_topic(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        tc = store.record_observation("fotónica", 0.8, source="search_corpus")
        assert tc.topic == "fotónica"
        assert tc.confidence == 0.8
        assert tc.observation_count == 1
        store.close()

    def test_record_observation_updates_existing(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        store.record_observation("topic1", 0.8)
        tc = store.record_observation("topic1", 0.4)
        # EMA: 0.8 * 0.7 + 0.4 * 0.3 = 0.56 + 0.12 = 0.68
        assert tc.observation_count == 2
        assert abs(tc.confidence - 0.68) < 0.01
        store.close()

    def test_low_confidence_flagged(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        # Two observations with low scores → needs_research
        store.record_observation("low_topic", 0.2)
        tc = store.record_observation("low_topic", 0.2)
        assert tc.observation_count >= MIN_OBSERVATIONS
        assert tc.confidence < CONFIDENCE_THRESHOLD
        assert tc.needs_research
        store.close()

    def test_high_confidence_not_flagged(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        store.record_observation("high_topic", 0.9)
        tc = store.record_observation("high_topic", 0.9)
        assert not tc.needs_research
        store.close()

    def test_list_low_confidence(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        for _ in range(3):
            store.record_observation("low1", 0.1)
        for _ in range(3):
            store.record_observation("low2", 0.2)
        for _ in range(3):
            store.record_observation("high1", 0.9)
        low = store.list_low_confidence()
        topics = [t.topic for t in low]
        assert "low1" in topics
        assert "low2" in topics
        assert "high1" not in topics
        store.close()

    def test_tracker_observe_search_zero_hits(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        tracker = UncertaintyTracker(store)
        tc = tracker.observe_search("query1", hits_count=0, avg_score=0.0)
        assert tc.confidence == 0.0
        store.close()

    def test_tracker_observe_search_few_hits(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        tracker = UncertaintyTracker(store)
        tc = tracker.observe_search("query2", hits_count=2, avg_score=0.8)
        # Few hits (<3) → score * 0.5
        assert abs(tc.last_score - 0.4) < 0.01
        store.close()

    def test_tracker_observe_compile(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        tracker = UncertaintyTracker(store)
        tc = tracker.observe_compile("topic", doc_count=0)
        assert tc.last_score == 0.0
        tc2 = tracker.observe_compile("topic2", doc_count=20)
        assert tc2.last_score == 0.85
        store.close()

    def test_active_research_agenda_proposes(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        # Create a low-confidence topic
        for _ in range(3):
            store.record_observation("mystery_topic", 0.1)
        agenda = ActiveResearchAgenda(store)
        proposals = agenda.scan_and_propose()
        assert len(proposals) >= 1
        p = proposals[0]
        assert p.topic == "mystery_topic"
        assert p.status == "pending"
        assert p.current_confidence < CONFIDENCE_THRESHOLD
        store.close()

    def test_active_research_agenda_dedupes(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        for _ in range(3):
            store.record_observation("dup_topic", 0.1)
        agenda = ActiveResearchAgenda(store)
        first = agenda.scan_and_propose()
        assert len(first) >= 1
        second = agenda.scan_and_propose()
        assert len(second) == 0
        store.close()

    def test_render_uncertainty_context_empty(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        assert render_uncertainty_context(store) == ""
        store.close()

    def test_render_uncertainty_context_content(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        for _ in range(3):
            store.record_observation("low_topic", 0.1)
        rendered = render_uncertainty_context(store)
        assert "low_topic" in rendered
        store.close()

    def test_decide_proposal(self, tmp_path):
        store = UncertaintyStore(tmp_path / "unc.db")
        for _ in range(3):
            store.record_observation("decide_topic", 0.1)
        agenda = ActiveResearchAgenda(store)
        proposals = agenda.scan_and_propose()
        store.decide_proposal(proposals[0].proposal_id, approved=True)
        loaded = store.list_proposals(status="approved", limit=10)
        assert len(loaded) == 1
        store.close()


# ---------------------------------------------------------------------------
# User model transversal (punto 8)
# ---------------------------------------------------------------------------

class TestUserModel:
    def test_add_and_list_goal(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.add_goal("paper sobre fotónica")
        goals = store.list_goals()
        assert len(goals) == 1
        assert goals[0].description == "paper sobre fotónica"
        assert goals[0].status == "active"
        store.close()

    def test_active_goals(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.add_goal("active goal")
        g2 = store.add_goal("goal to abandon", source="inferred", confidence=0.5)
        store.decide_goal(g2.goal_id, approved=False)
        active = store.active_goals()
        assert len(active) == 1
        assert active[0].description == "active goal"
        store.close()

    def test_record_interest_observation(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        for _ in range(5):
            store.record_interest_observation("retrieval")
        interests = store.list_interests()
        assert len(interests) == 1
        assert interests[0].topic == "retrieval"
        assert interests[0].occurrence_count == 5
        assert interests[0].score > 0.0
        store.close()

    def test_declare_interest_high_score(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.declare_interest("rust")
        interests = store.list_interests()
        assert len(interests) == 1
        assert interests[0].score == 1.0
        assert interests[0].source == "declared"
        store.close()

    def test_set_and_get_preference(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.set_preference("response_length", "short")
        pref = store.get_preference("response_length")
        assert pref is not None
        assert pref.value == "short"
        store.close()

    def test_add_and_list_fact(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.add_fact("prefiere respuestas con citas", status="active")
        facts = store.active_facts()
        assert len(facts) == 1
        assert "citas" in facts[0]
        store.close()

    def test_render_user_model_empty(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        assert render_user_model_context(store) == ""
        store.close()

    def test_render_user_model_with_data(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        store.add_goal("paper sobre fotónica")
        store.declare_interest("fotónica")
        store.set_preference("response_length", "short")
        store.add_fact("trabaja en IPA", status="active")
        rendered = render_user_model_context(store)
        assert "paper sobre fotónica" in rendered
        assert "fotónica" in rendered
        assert "short" in rendered or "cort" in rendered.lower()
        store.close()

    def test_inferer_interests_from_episodes(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        inferer = UserModelInferer(store)
        episodes = [
            {"turn_role": "assistant", "content": "[TOOL:search_corpus]{\"query\": \"retrieval\"}"},
            {"turn_role": "assistant", "content": "[TOOL:search_corpus]{\"query\": \"retrieval\"}"},
            {"turn_role": "assistant", "content": "[TOOL:search_corpus]{\"query\": \"retrieval\"}"},
            {"turn_role": "assistant", "content": "[TOOL:search_corpus]{\"query\": \"chunking\"}"},
        ]
        updated = inferer.infer_interests_from_episodes(episodes)
        # "retrieval" appears 3 times (>= MIN_INTEREST_OCCURRENCES)
        topics = [u.topic for u in updated]
        assert "retrieval" in topics
        store.close()

    def test_inferer_goals_from_tasks(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        inferer = UserModelInferer(store)
        tasks = [
            {"goal": "investigá fotónica"},
            {"goal": "investigá fotónica aplicada"},
            {"goal": "hacé un reporte sobre rust"},
        ]
        proposals = inferer.infer_goals_from_tasks(tasks)
        # "fotónica" appears in 2 tasks (>= MIN_TASKS_FOR_GOAL)
        assert len(proposals) >= 1
        assert any("fotónica" in p.description for p in proposals)
        store.close()

    def test_inferer_response_length_preference(self, tmp_path):
        store = UserModelStore(tmp_path / "um.db")
        inferer = UserModelInferer(store)
        episodes = []
        for _ in range(4):
            episodes.append({"turn_role": "user", "content": "más detalle por favor"})
            episodes.append({"turn_role": "assistant", "content": "respuesta"})
        pref = inferer.infer_response_length_preference(episodes)
        assert pref is not None
        assert pref.value == "long"
        store.close()


# ---------------------------------------------------------------------------
# Integration: system prompt injection
# ---------------------------------------------------------------------------

class TestSystemPromptInjection:
    def test_system_prompt_includes_user_model_when_data_exists(self, tmp_path, monkeypatch):
        # Point stores to temp
        um_path = tmp_path / "um.db"
        monkeypatch.setattr("ipa.agent.user_model.DEFAULT_USER_MODEL_STORE", um_path)
        store = UserModelStore(um_path)
        store.add_goal("test goal")
        store.declare_interest("test interest")
        store.close()
        from ipa.agent.agent_identity import load_identity
        identity = load_identity()
        prompt = identity.system_prompt()
        assert "test goal" in prompt or "test interest" in prompt

    def test_system_prompt_empty_when_no_data(self, tmp_path, monkeypatch):
        # Use empty temp stores
        monkeypatch.setattr("ipa.agent.user_model.DEFAULT_USER_MODEL_STORE", tmp_path / "empty_um.db")
        monkeypatch.setattr("ipa.agent.strategic_memory.DEFAULT_STRATEGIC_STORE", tmp_path / "empty_strat.db")
        monkeypatch.setattr("ipa.agent.skill_library.DEFAULT_SKILL_STORE", tmp_path / "empty_skill.db")
        monkeypatch.setattr("ipa.agent.uncertainty.DEFAULT_UNCERTAINTY_STORE", tmp_path / "empty_unc.db")
        from ipa.agent.agent_identity import load_identity
        identity = load_identity()
        prompt = identity.system_prompt()
        # Should not contain dynamic context sections
        assert "Perfil del usuario" not in prompt
        assert "Principios estratégicos" not in prompt
        assert "Skills aprendidas" not in prompt
        assert "baja confianza" not in prompt

    def test_system_prompt_include_user_model_false(self):
        from ipa.agent.agent_identity import load_identity
        identity = load_identity()
        prompt = identity.system_prompt(include_user_model=False)
        assert "Perfil del usuario" not in prompt


# ---------------------------------------------------------------------------
# System tools integration (puntos 2-6, 8)
# ---------------------------------------------------------------------------

class TestSystemToolsCognitive:
    def test_plan_task_in_catalog(self):
        from ipa.agent.system_tools import TOOL_CATALOG
        assert "plan_task" in TOOL_CATALOG

    def test_list_tasks_in_catalog(self):
        from ipa.agent.system_tools import TOOL_CATALOG
        assert "list_tasks" in TOOL_CATALOG

    def test_get_user_profile_in_catalog(self):
        from ipa.agent.system_tools import TOOL_CATALOG
        assert "get_user_profile" in TOOL_CATALOG

    def test_list_research_agenda_in_catalog(self):
        from ipa.agent.system_tools import TOOL_CATALOG, SYSTEM_TOOL_NAMES
        # list_research_agenda is now hidden from the catalog (reduced cognitive
        # load for the 9B) but still dispatchable internally.
        assert "list_research_agenda" in SYSTEM_TOOL_NAMES

    def test_set_user_goal_in_catalog(self):
        from ipa.agent.system_tools import TOOL_CATALOG
        assert "set_user_goal" in TOOL_CATALOG

    def test_execute_list_tasks(self, tmp_path, monkeypatch):
        from ipa.agent.system_tools import execute_system_tool
        from ipa.agent.task_planner import TaskStore
        monkeypatch.setattr("ipa.agent.task_planner.DEFAULT_TASK_STORE", tmp_path / "task.db")
        # Patch the tool to use the temp store
        import ipa.agent.system_tools as st
        original = st.tool_list_tasks
        def patched(args):
            from ipa.agent.task_planner import TaskStore
            store = TaskStore(tmp_path / "task.db")
            try:
                tasks = store.list_tasks(limit=args.get("limit", 10))
                return type(original(args))(
                    tool_name="list_tasks", ok=True,
                    summary=f"{len(tasks)} tasks", data={"tasks": [], "total": len(tasks), "active": 0},
                )
            finally:
                store.close()
        # Just verify the tool is dispatchable
        result = execute_system_tool("list_tasks", {})
        assert result.ok
        assert "tasks" in result.data or "total" in result.data

    def test_execute_get_user_profile(self):
        from ipa.agent.system_tools import execute_system_tool
        result = execute_system_tool("get_user_profile", {})
        assert result.ok
        assert "goals" in result.data

    def test_execute_list_research_agenda(self):
        from ipa.agent.system_tools import execute_system_tool
        result = execute_system_tool("list_research_agenda", {})
        assert result.ok
        assert "low_confidence_topics" in result.data

    def test_execute_set_user_goal(self, tmp_path, monkeypatch):
        from ipa.agent.system_tools import execute_system_tool
        from ipa.agent.user_model import UserModelStore
        monkeypatch.setattr("ipa.agent.user_model.DEFAULT_USER_MODEL_STORE", tmp_path / "um.db")
        result = execute_system_tool("set_user_goal", {"description": "test goal"})
        assert result.ok
        assert "test goal" in result.summary

    def test_execute_set_user_interest(self, tmp_path, monkeypatch):
        from ipa.agent.system_tools import execute_system_tool
        monkeypatch.setattr("ipa.agent.user_model.DEFAULT_USER_MODEL_STORE", tmp_path / "um.db")
        result = execute_system_tool("set_user_interest", {"topic": "test interest"})
        assert result.ok

    def test_plan_task_validates_goal(self):
        from ipa.agent.system_tools import execute_system_tool
        result = execute_system_tool("plan_task", {})
        assert not result.ok
        assert "goal" in (result.error or "").lower()

    def test_get_task_validates_task_id(self):
        from ipa.agent.system_tools import execute_system_tool
        result = execute_system_tool("get_task", {})
        assert not result.ok
        assert "task_id" in (result.error or "").lower()

    def test_resume_task_validates_task_id(self):
        from ipa.agent.system_tools import execute_system_tool
        result = execute_system_tool("resume_task", {})
        assert not result.ok
        assert "task_id" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Async tool wait (fix bug de timing 2026-09-09)
# ---------------------------------------------------------------------------

class TestAsyncToolWait:
    def test_async_tools_set_contains_research_and_ingestion(self):
        from ipa.agent.task_planner import _ASYNC_TOOLS
        assert "research_topic" in _ASYNC_TOOLS
        assert "run_ingestion" in _ASYNC_TOOLS

    def test_async_timeout_research_uses_max_seconds(self):
        from ipa.agent.task_planner import _async_timeout
        # max_seconds=120 → timeout=180 (120+60 buffer)
        t = _async_timeout("research_topic", {"max_seconds": 120})
        assert t == 180

    def test_async_timeout_research_capped(self):
        from ipa.agent.task_planner import _async_timeout
        # max_seconds=400 → timeout=460, but capped at 400
        t = _async_timeout("research_topic", {"max_seconds": 400})
        assert t == 400

    def test_async_timeout_ingestion_default(self):
        from ipa.agent.task_planner import _async_timeout
        t = _async_timeout("run_ingestion", {})
        assert t == 600

    def test_wait_for_async_completion_timeout(self, tmp_path, monkeypatch):
        """Si el progress file nunca cambia de 'running', el wait hace timeout."""
        from ipa.agent.task_planner import _wait_for_async_completion
        import ipa.agent.system_tools as st

        # Mock _read_progress to always return "running"
        def mock_read(name):
            return {"status": "running"}
        monkeypatch.setattr(st, "_read_progress", mock_read)

        # Short timeout for test speed
        result = _wait_for_async_completion("research_topic", {"max_seconds": 30}, timeout=2)
        assert not result["ok"]
        assert result["status"] == "timeout"

    def test_wait_for_async_completion_done(self, tmp_path, monkeypatch):
        """Si el progress file cambia a 'done', el wait retorna ok=True."""
        from ipa.agent.task_planner import _wait_for_async_completion
        import ipa.agent.system_tools as st

        call_count = [0]
        def mock_read(name):
            call_count[0] += 1
            # After 2 polls, return done
            if call_count[0] < 2:
                return {"status": "running"}
            return {
                "status": "done",
                "result": {"search_results": 10, "scraped": 5, "ingested": 3},
            }
        monkeypatch.setattr(st, "_read_progress", mock_read)

        result = _wait_for_async_completion("research_topic", {"max_seconds": 30}, timeout=30)
        assert result["ok"]
        assert result["status"] == "done"
        assert "3 ingestado" in result["summary"]

    def test_wait_for_async_completion_failed(self, tmp_path, monkeypatch):
        """Si el progress file cambia a 'failed', el wait retorna ok=False."""
        from ipa.agent.task_planner import _wait_for_async_completion
        import ipa.agent.system_tools as st

        call_count = [0]
        def mock_read(name):
            call_count[0] += 1
            if call_count[0] < 2:
                return {"status": "running"}
            return {"status": "failed", "error": "network error"}
        monkeypatch.setattr(st, "_read_progress", mock_read)

        result = _wait_for_async_completion("research_topic", {"max_seconds": 30}, timeout=30)
        assert not result["ok"]
        assert result["status"] == "failed"
        assert "network error" in result["error"]
