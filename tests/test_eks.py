from __future__ import annotations

from pathlib import Path

import pytest

from tools.eks_repository import EKSRepository

REPO_ROOT = Path(__file__).resolve().parents[1]


VALID_DECISION = """---
id: DEC-001
category: decision
status: accepted
created: 2026-09-05
updated: 2026-09-05
author: test
components: [eks, mcp]
tags: [read-only, context]
related: []
supersedes: null
superseded_by: null
---

# DEC-001 — Read-only EKS context

A deterministic context boundary for development.
"""


def _repo(tmp_path: Path, content: str = VALID_DECISION) -> EKSRepository:
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True, exist_ok=True)
    (root / "_schema").mkdir(exist_ok=True)
    (root / "README.md").write_text("# EKS\n", encoding="utf-8")
    (root / "decisions" / "DEC-001.md").write_text(content, encoding="utf-8")
    return EKSRepository(root)


def test_empty_repository_is_valid_and_does_not_require_content(tmp_path: Path):
    root = tmp_path / "knowledge"
    root.mkdir()
    report = EKSRepository(root).validate()
    assert report.valid
    assert report.records == ()


def test_valid_record_is_catalogued_and_searchable(tmp_path: Path):
    repository = _repo(tmp_path)
    report = repository.validate()
    assert report.valid
    assert [record.record_id for record in report.records] == ["DEC-001"]
    results = repository.search("read-only context", component="mcp")
    assert results[0]["id"] == "DEC-001"
    assert results[0]["score"] > 0
    assert results[0]["matched_fields"]


def test_validation_rejects_duplicate_or_mismatched_metadata(tmp_path: Path):
    content = VALID_DECISION.replace("id: DEC-001", "id: EXP-001")
    repository = _repo(tmp_path, content)
    report = repository.validate()
    assert not report.valid
    assert any("id/category mismatch" in error for error in report.errors)


def test_get_rejects_paths_outside_allowed_roots(tmp_path: Path):
    repository = _repo(tmp_path)
    with pytest.raises(KeyError):
        repository.get("../secret.md")
    with pytest.raises(KeyError):
        repository.get(str(tmp_path / "secret.md"))


def test_context_is_honest_when_no_knowledge_matches(tmp_path: Path):
    repository = EKSRepository(tmp_path / "knowledge")
    (tmp_path / "knowledge").mkdir()
    context = repository.context("unknown task")
    assert context["knowledge_available"] is False
    assert context["results"] == []
    assert context["notice"]


def test_real_knowledge_tree_is_valid():
    """The committed knowledge/ tree must pass validation — not only synthetic
    fixtures. Warnings (stale artifact links, component aliases) are allowed;
    errors are not."""
    report = EKSRepository(REPO_ROOT / "knowledge").validate()
    assert report.valid, "\n".join(report.errors)


def test_real_tree_has_minimal_coverage():
    repo = EKSRepository(REPO_ROOT / "knowledge")
    report = repo.report()
    assert report["total_records"] >= 30
    for category in ("decision", "experiment", "benchmark", "postmortem", "pattern", "research"):
        assert report["by_category"][category] >= 1


def test_updated_must_not_predate_created(tmp_path: Path):
    content = VALID_DECISION.replace("updated: 2026-09-05", "updated: 2026-09-04")
    report = _repo(tmp_path, content).validate()
    assert not report.valid
    assert any("predates created" in error for error in report.errors)


def test_superseded_requires_superseded_by(tmp_path: Path):
    content = VALID_DECISION.replace("status: accepted", "status: superseded")
    report = _repo(tmp_path, content).validate()
    assert not report.valid
    assert any("superseded_by" in error for error in report.errors)


def test_supersedes_pair_must_be_reciprocal(tmp_path: Path):
    old = VALID_DECISION.replace("status: accepted", "status: superseded").replace(
        "superseded_by: null", "superseded_by: DEC-002")
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    (root / "decisions" / "DEC-001.md").write_text(old, encoding="utf-8")
    # DEC-002 forgets to declare supersedes DEC-001 → warning, not error.
    new = VALID_DECISION.replace("id: DEC-001", "id: DEC-002")
    (root / "decisions" / "DEC-002.md").write_text(new, encoding="utf-8")
    report = EKSRepository(root).validate()
    assert report.valid
    assert any("does not declare supersedes DEC-001" in w for w in report.warnings)


def test_component_vocabulary_warns_on_alias_and_unknown(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "_schema").mkdir(parents=True)
    (root / "_schema" / "components.json").write_text(
        '{"components": ["eks"], "aliases": {"eks_old": "eks"}}', encoding="utf-8")
    (root / "decisions").mkdir()
    content = VALID_DECISION.replace(
        "components: [eks, mcp]", "components: [eks_old, mystery_box]")
    (root / "decisions" / "DEC-001.md").write_text(content, encoding="utf-8")
    report = EKSRepository(root).validate()
    assert report.valid
    assert any("alias of 'eks'" in w for w in report.warnings)
    assert any("unknown component 'mystery_box'" in w for w in report.warnings)


def test_missing_artifact_links_warn_only(tmp_path: Path):
    content = VALID_DECISION + "\nEvidence in outputs/experiments/E99/report.json\n"
    report = _repo(tmp_path, content).validate()
    assert report.valid
    assert any("cited artifact missing" in w for w in report.warnings)


def test_search_weights_title_over_body(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    title_hit = VALID_DECISION.replace(
        "# DEC-001 — Read-only EKS context", "# DEC-001 — Vector index selection")
    body_hit = VALID_DECISION.replace("id: DEC-001", "id: DEC-002").replace(
        "A deterministic context boundary", "vector index selection discussion")
    (root / "decisions" / "DEC-001.md").write_text(title_hit, encoding="utf-8")
    (root / "decisions" / "DEC-002.md").write_text(body_hit, encoding="utf-8")
    results = EKSRepository(root).search("vector index selection")
    assert results[0]["id"] == "DEC-001"
    assert "title" in results[0]["matched_fields"]


def test_context_excludes_closed_records_by_default(tmp_path: Path):
    rejected = VALID_DECISION.replace("status: accepted", "status: rejected")
    repo = _repo(tmp_path, rejected)
    assert repo.context("read-only context")["knowledge_available"] is False
    assert repo.context("read-only context", include_historical=True)["results"]


def test_context_expands_related_one_hop(tmp_path: Path):
    root = tmp_path / "knowledge"
    for folder in ("decisions", "patterns"):
        (root / folder).mkdir(parents=True)
    decision = VALID_DECISION.replace("related: []", "related: [PAT-001]")
    # The pattern does not match the query terms — it only surfaces via the
    # 1-hop `related` expansion from DEC-001.
    pattern = (
        VALID_DECISION.replace("id: DEC-001", "id: PAT-001")
        .replace("category: decision", "category: pattern")
        .replace("tags: [read-only, context]", "tags: [storage]")
        .replace("# DEC-001 — Read-only EKS context", "# PAT-001 — Storage layout")
        .replace("A deterministic context boundary for development.",
                 "Canonical persistence layout.")
    )
    (root / "decisions" / "DEC-001.md").write_text(decision, encoding="utf-8")
    (root / "patterns" / "PAT-001.md").write_text(pattern, encoding="utf-8")
    context = EKSRepository(root).context("read-only context")
    assert [item["id"] for item in context["related"]] == ["PAT-001"]
    assert context["related"][0]["via"] == "DEC-001"


def test_context_relaxes_strict_filters_when_starved(tmp_path: Path):
    repo = _repo(tmp_path)  # record has components [eks, mcp]
    strict = repo.context("read-only context", components=["eks", "retrieval"])
    assert strict["filters_relaxed"] is True
    assert strict["results"]


def test_scaffold_assigns_next_id_and_valid_frontmatter(tmp_path: Path):
    from tools.eks_scaffold import scaffold

    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    (root / "_templates").mkdir()
    (root / "decisions" / "DEC-007-old.md").write_text(VALID_DECISION, encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "eks_scaffold.py").write_text("# scaffold\n")
    path = scaffold(root, "decision", "My new rule", status="proposed",
                    components=["eks"], tags=["governance"],
                    affects=["src/ipa/**"], evidence=["tools/eks_scaffold.py"],
                    author_model="swe-2", trigger="permit:PW-1")
    assert path.name.startswith("DEC-008-")
    text = path.read_text(encoding="utf-8")
    assert "affects: [src/ipa/**]" in text
    assert "author_model: swe-2" in text
    report = EKSRepository(root).validate()
    assert report.valid, report.errors


# --- affects / governing --------------------------------------------------

def test_glob_match_semantics():
    from tools.eks_repository import glob_match
    assert glob_match("src/ipa/agentic/x.py", "src/ipa/agentic/**")
    assert glob_match("src/ipa/agentic/x.py", "src/**/*.py")
    assert not glob_match("src/ipa/agentic/x.py", "src/*.py")  # * no cruza /
    assert glob_match("docs/USAGE.md", "docs/**")
    assert not glob_match("knowledge/x.md", "docs/**")
    assert glob_match("outputs/agent/vram.lock", "outputs/agent/*.lock")
    assert not glob_match("outputs/agent/deep/vram.lock", "outputs/agent/*.lock")


def test_governing_matches_affects_and_includes_closed(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    (root / "patterns").mkdir(parents=True)
    decision = VALID_DECISION.replace(
        "superseded_by: null",
        'superseded_by: null\naffects: ["src/ipa/agentic/**"]')
    (root / "decisions" / "DEC-001.md").write_text(decision, encoding="utf-8")
    rejected = (
        VALID_DECISION.replace("id: DEC-001", "id: PAT-001")
        .replace("category: decision", "category: pattern")
        .replace("status: accepted", "status: rejected")
        .replace("superseded_by: null",
                 'superseded_by: null\naffects: ["src/ipa/agentic/**"]')
    )
    (root / "patterns" / "PAT-001.md").write_text(rejected, encoding="utf-8")
    repo = EKSRepository(root)
    hits = repo.governing(["src/ipa/agentic/promotion_executor.py"])
    assert {item["id"] for item in hits} == {"DEC-001", "PAT-001"}
    assert repo.governing(["docs/USAGE.md"]) == []
    live = repo.governing(["src/ipa/agentic/x.py"], include_closed=False)
    assert [item["id"] for item in live] == ["DEC-001"]


def test_context_paths_activate_applicable_section(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    (root / "patterns").mkdir(parents=True)
    decision = VALID_DECISION.replace(
        "superseded_by: null",
        'superseded_by: null\naffects: ["src/ipa/**"]')
    (root / "decisions" / "DEC-001.md").write_text(decision, encoding="utf-8")
    pattern = (
        VALID_DECISION.replace("id: DEC-001", "id: PAT-001")
        .replace("category: decision", "category: pattern")
        .replace("tags: [read-only, context]", "tags: [unrelated]")
        .replace("# DEC-001 — Read-only EKS context", "# PAT-001 — Other")
        .replace("A deterministic context boundary for development.", "x")
        .replace("superseded_by: null",
                 'superseded_by: null\naffects: ["src/ipa/**"]')
    )
    (root / "patterns" / "PAT-001.md").write_text(pattern, encoding="utf-8")
    context = EKSRepository(root).context("zzz no-match query",
                                        paths=["src/ipa/foo.py"])
    assert {item["id"] for item in context["applicable"]} == {"DEC-001", "PAT-001"}
    assert context["knowledge_available"] is True


# --- governance gates -----------------------------------------------------

def _new_accepted(content: str) -> str:
    """VALID_DECISION as a post-cutoff record (created today)."""
    return (content
            .replace("created: 2026-09-05", "created: 2026-09-23")
            .replace("updated: 2026-09-05", "updated: 2026-09-23"))


def test_accepted_new_record_requires_verifiable_evidence(tmp_path: Path):
    repo = _repo(tmp_path, _new_accepted(VALID_DECISION))
    report = repo.validate()
    assert not report.valid
    assert any("no verifiable evidence" in e for e in report.errors)


def test_evidence_field_satisfies_gate_and_must_exist(tmp_path: Path):
    base = VALID_DECISION.replace(
        "superseded_by: null",
        'superseded_by: null\nevidence: ["knowledge/decisions/DEC-001.md"]')
    repo = _repo(tmp_path, _new_accepted(base))
    assert repo.validate().valid  # evidence resolves under tmp root
    missing = _new_accepted(base).replace(
        'evidence: ["knowledge/decisions/DEC-001.md"]', 'evidence: ["src/nope.py"]')
    report = _repo(tmp_path, missing).validate()
    assert not report.valid
    assert any("evidence path missing" in e for e in report.errors)


def test_existing_body_citation_satisfies_gate(tmp_path: Path):
    body = VALID_DECISION.replace(
        "A deterministic context boundary for development.",
        "See outputs/experiments/E0/report.json for results.")
    (tmp_path / "outputs" / "experiments" / "E0").mkdir(parents=True)
    (tmp_path / "outputs" / "experiments" / "E0" / "report.json").write_text("{}")
    repo = _repo(tmp_path, _new_accepted(body))
    assert repo.validate().valid


def test_agent_author_needs_author_model_after_cutoff(tmp_path: Path):
    record = _new_accepted(VALID_DECISION).replace("author: test", "author: agent")
    record = record.replace(
        "superseded_by: null",
        'superseded_by: null\nevidence: ["knowledge/decisions/DEC-001.md"]')
    report = _repo(tmp_path, record).validate()
    assert any("author_model" in w for w in report.warnings)


def test_affects_glob_must_match_repo_files(tmp_path: Path):
    record = VALID_DECISION.replace(
        "superseded_by: null",
        'superseded_by: null\naffects: ["src/nonexistent_dir/**", "outputs/agent/*.lock"]')
    report = _repo(tmp_path, record).validate()
    assert any("affects glob matches no files" in w for w in report.warnings)
    # outputs/ globs are runtime-transient — exempt from liveness.
    assert not any("outputs/agent/*.lock" in w and "matches no files" in w
                   for w in report.warnings)


# --- work permits ---------------------------------------------------------

def test_permit_lifecycle_and_conflict(tmp_path: Path):
    from tools.work_permits import PermitStore
    store = PermitStore(tmp_path / "permits")
    permit, payload = store.acquire(
        "s1", ["src/ipa/**"], "refactor", type="exclusive", ttl_s=60)
    assert payload["issued"] and permit.permit_id.startswith("PW-")
    # Overlapping exclusive scope from another session is refused.
    _, denied = store.acquire("s2", ["src/ipa/agentic/**"], "collide")
    assert denied["issued"] is False and denied["conflicts"]
    # Same session may take a nested permit; disjoint scope is fine.
    other, ok = store.acquire("s2", ["docs/**"], "docs")
    assert ok["issued"]
    # check_path: foreign session blocked, holder allowed. s2's docs/**
    # permit symmetrically blocks s1 there.
    assert store.check_path("src/ipa/x.py", session="s2")["blocked"]
    assert not store.check_path("src/ipa/x.py", session="s1")["blocked"]
    assert not store.check_path("docs/x.md", session="s2")["blocked"]
    assert store.check_path("docs/x.md", session="s1")["blocked"]
    # Expiry: TTL in the past kills only that permit; the other stays live.
    permit.ttl_s = -1
    store._save(permit)
    assert [p.permit_id for p in store.active()] == [other.permit_id]
    store.close_session("s1")  # idempotent on expired


def test_permit_acquire_attaches_governing_precautions(tmp_path: Path):
    from tools.work_permits import PermitStore
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    decision = VALID_DECISION.replace(
        "superseded_by: null",
        'superseded_by: null\naffects: ["src/ipa/agentic/**"]')
    (root / "decisions" / "DEC-001.md").write_text(decision, encoding="utf-8")
    store = PermitStore(tmp_path / "permits", EKSRepository(root))
    permit, payload = store.acquire("s1", ["src/ipa/agentic/x.py"], "work")
    assert payload["issued"]
    assert payload["precautions"] == ["DEC-001"]
    assert permit.precautions == ["DEC-001"]


def test_permit_acquire_is_atomic_under_concurrency(tmp_path: Path):
    """The check-then-write of `acquire` runs under one cross-process lock:
    eight sessions racing for the same exclusive scope → exactly one wins."""
    from concurrent.futures import ThreadPoolExecutor
    from tools.work_permits import PermitStore

    store = PermitStore(tmp_path / "permits")

    def attempt(index: int) -> bool:
        _permit, payload = store.acquire(f"s{index}", ["src/ipa/**"], "collide")
        return payload["issued"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        issued = list(pool.map(attempt, range(8)))

    assert sum(issued) == 1, issued
    assert len(store.active()) == 1
    assert not (tmp_path / "permits" / ".acquire.lock").exists()


# --- component vocabulary (groups / aliases) ------------------------------

def _component_repo(tmp_path: Path) -> EKSRepository:
    root = tmp_path / "knowledge"
    (root / "_schema").mkdir(parents=True)
    (root / "_schema" / "components.json").write_text(
        '{"components": ["indexes", "vector_index", "lexical_index", "dashboard"],'
        ' "aliases": {"dashboard_api": "dashboard"},'
        ' "groups": {"indexes": ["lexical_index", "vector_index"]}}',
        encoding="utf-8")
    (root / "decisions").mkdir(parents=True)
    (root / "decisions" / "DEC-001.md").write_text(
        VALID_DECISION.replace("components: [eks, mcp]", "components: [vector_index]")
        .replace("# DEC-001 — Read-only EKS context", "# DEC-001 — Vector index choice"),
        encoding="utf-8")
    (root / "decisions" / "DEC-002.md").write_text(
        VALID_DECISION.replace("id: DEC-001", "id: DEC-002")
        .replace("components: [eks, mcp]", "components: [indexes]")
        .replace("# DEC-001 — Read-only EKS context", "# DEC-001 — Index umbrella"),
        encoding="utf-8")
    (root / "decisions" / "DEC-003.md").write_text(
        VALID_DECISION.replace("id: DEC-001", "id: DEC-003")
        .replace("components: [eks, mcp]", "components: [dashboard]")
        .replace("# DEC-001 — Read-only EKS context", "# DEC-001 — Dashboard"),
        encoding="utf-8")
    return EKSRepository(root)


def test_component_group_filter_expands_both_directions(tmp_path: Path):
    repository = _component_repo(tmp_path)
    by_group = repository.search("index", component="indexes")
    assert {item["id"] for item in by_group} == {"DEC-001", "DEC-002"}
    by_member = repository.search("index", component="vector_index")
    assert {item["id"] for item in by_member} == {"DEC-001", "DEC-002"}


def test_component_alias_resolves_in_filter(tmp_path: Path):
    repository = _component_repo(tmp_path)
    assert [item["id"] for item in repository.search("dashboard", component="dashboard_api")] \
        == ["DEC-003"]
    assert repository.component_matches({"dashboard"}, "dashboard_api")
    assert not repository.component_matches({"dashboard"}, "indexes")


# --- hot zones ------------------------------------------------------------

def test_report_exposes_overlap_hot_zones_beyond_exact_globs(tmp_path: Path):
    """Four records governing distinct files in the same directory: no exact
    glob repeats, but the prefix-overlap view (same criterion as permit
    precautions) must flag the directory."""
    root = tmp_path / "knowledge"
    (root / "patterns").mkdir(parents=True)
    for index, name in enumerate(("promotion_executor.py", "promotion_policy.py",
                                  "tier0.py", "index_audit.py"), start=1):
        record = (VALID_DECISION
                  .replace("id: DEC-001", f"id: PAT-{index:03d}")
                  .replace("category: decision", "category: pattern")
                  .replace("superseded_by: null",
                           f'superseded_by: null\naffects: ["src/ipa/agentic/{name}"]'))
        (root / "patterns" / f"PAT-{index:03d}.md").write_text(record, encoding="utf-8")

    report = EKSRepository(root).report()
    assert report["hot_zones"] == {}
    assert report["hot_zones_overlap"]["src/ipa/agentic/"] == [
        "PAT-001", "PAT-002", "PAT-003", "PAT-004"]


# --- liveness scan --------------------------------------------------------

def test_liveness_scan_skips_runtime_trees(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "decisions").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "outputs" / "big").mkdir(parents=True)
    (tmp_path / "outputs" / "big" / "artifact.txt").write_text("x", encoding="utf-8")
    (tmp_path / "Archive").mkdir()
    (tmp_path / "Archive" / "old.txt").write_text("x", encoding="utf-8")

    files = EKSRepository(root)._repo_files()
    assert "src/a.py" in files
    assert not any(item.startswith("outputs/") for item in files)
    assert not any(item.startswith("Archive/") for item in files)
