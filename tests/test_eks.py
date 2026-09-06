from __future__ import annotations

from pathlib import Path

import pytest

from tools.eks_repository import EKSRepository


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
    (root / "decisions").mkdir(parents=True)
    (root / "_schema").mkdir()
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
    assert results[0]["score"] == 2


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
