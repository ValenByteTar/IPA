import json
from pathlib import Path

CONTRACT = Path(__file__).parents[1] / "contracts" / "contract_vocabulary.json"

EXPECTED_RECORDS = {
    "ArtifactRef",
    "ProcessingManifest",
    "ParserResult",
    "CanonicalDocument",
    "DocumentChunk",
    "SourceSpan",
    "EmbeddingRecord",
    "SearchHit",
    "KnowledgeDelta",
    "KnowledgeState",
    "LearningGoal",
    "Concept",
    "Roadmap",
    "AssessmentResult",
    "ResearchRequest",
    "ReporterReport",
    "ReporterDocumentDecision",
    "TopicLink",
    "AgentSession",
    "AgentEpisode",
    "ToolCall",
    "ToolResult",
    "WebSource",
    "UserTopicRecord",
    "UserEvidence",
    "TopicCluster",
}

EXPECTED_INVARIANTS = {
    "original_artifact_is_immutable",
    "every_derived_record_has_input_hash",
    "every_search_hit_has_provenance",
    "external_tools_do_not_write_authoritative_state",
    "base_lexical_index_can_exist_without_embeddings",
    "generated_tutor_fields_have_generation_provenance",
    "source_backed_tutor_fields_have_source_references",
    "confirmed_goals_require_human_approval",
    "active_roadmaps_require_human_approval",
    "research_execution_requires_human_approval",
    "mastery_is_supported_by_assessment_evidence",
    "user_topic_records_require_evidence_for_mastery",
    "user_evidence_is_append_only",
    "topic_clusters_are_emergent",
    "topic_clusters_are_derived_not_authoritative",
    "memory_consolidation_requires_human_approval",
    "reporter_categories_are_emergent",
    "reporter_rejections_do_not_delete_artifacts",
    "reporter_promotion_requires_human_approval",
    "reporter_reports_are_corpus_fingerprinted",
    "agent_episodes_are_append_only",
    "agent_memory_lives_outside_any_corpus",
    "agent_episodes_record_active_identity_hash",
    "tool_calls_are_deterministic",
    "tool_results_record_content_hash",
    "web_sources_never_become_canonical",
    "web_sources_require_trust_label",
}

# Cross-record field expectations: every record must carry identity + provenance.
REQUIRED_FIELDS_EVERY_RECORD = {"artifact_id"} | {
    "document_id",  # CanonicalDocument / DocumentChunk
    "chunk_id",     # DocumentChunk / EmbeddingRecord / SearchHit
    "delta_id",     # KnowledgeDelta
    "state_version",  # KnowledgeState
}


def _load():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    return data


def test_contract_vocabulary_has_all_authoritative_records():
    data = _load()
    records = set(data["records"].keys())
    assert records == EXPECTED_RECORDS, f"missing/extra records: {records ^ EXPECTED_RECORDS}"


def test_contract_vocabulary_has_all_invariants():
    data = _load()
    invariants = set(data["invariants"])
    assert invariants == EXPECTED_INVARIANTS, (
        f"missing/extra invariants: {invariants ^ EXPECTED_INVARIANTS}"
    )


def test_artifact_ref_includes_landing_provenance_fields():
    """ArtifactRef must carry the provenance fields the landing manifest produces."""
    data = _load()
    fields = set(data["records"]["ArtifactRef"])
    required = {"artifact_id", "content_hash", "source_uri", "mime_type",
                "original_filename", "byte_size", "received_at"}
    assert required <= fields, f"ArtifactRef missing: {required - fields}"


def test_processing_manifest_declares_attempts_and_stages():
    data = _load()
    fields = set(data["records"]["ProcessingManifest"])
    assert {"artifact_id", "stages", "status", "attempts"} <= fields


def test_every_record_declares_artifact_id_or_equivalent_identity():
    """Every record must have an identity field; most carry artifact_id for provenance."""
    data = _load()
    identity_fields = {
        "ArtifactRef": "artifact_id",
        "ProcessingManifest": "artifact_id",
        "ParserResult": "artifact_id",
        "CanonicalDocument": "document_id",
        "DocumentChunk": "chunk_id",
        "SourceSpan": "artifact_id",
        "EmbeddingRecord": "chunk_id",
        "SearchHit": "chunk_id",
        "KnowledgeDelta": "delta_id",
        "KnowledgeState": "state_version",
        "LearningGoal": "goal_id",
        "Concept": "concept_id",
        "Roadmap": "roadmap_id",
        "AssessmentResult": "assessment_id",
        "ResearchRequest": "request_id",
        "AgentSession": "session_id",
        "AgentEpisode": "episode_id",
        "ToolCall": "tool_call_id",
        "ToolResult": "tool_result_id",
        "WebSource": "web_source_id",
        "UserTopicRecord": "record_id",
        "UserEvidence": "evidence_id",
        "TopicCluster": "cluster_id",
    }
    for record_name, identity in identity_fields.items():
        assert identity in data["records"][record_name], (
            f"{record_name} missing identity field '{identity}'"
        )
