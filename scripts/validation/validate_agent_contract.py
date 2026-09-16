"""Validate Agent core records against their authoritative JSON Schemas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
CONTRACTS = ROOT / "contracts"
SCHEMAS = {
    "AgentSession": "agent_session.schema.json",
    "AgentEpisode": "agent_episode.schema.json",
    "ToolCall": "tool_call.schema.json",
    "ToolResult": "tool_result.schema.json",
    "WebSource": "web_source.schema.json",
    "UserTopicRecord": "user_topic_record.schema.json",
    "UserEvidence": "user_evidence.schema.json",
    "TopicCluster": "topic_cluster.schema.json",
}


def load_schema(record_type: str) -> dict[str, Any]:
    if record_type not in SCHEMAS:
        raise ValueError(f"Unknown record type: {record_type}")
    return json.loads((CONTRACTS / SCHEMAS[record_type]).read_text(encoding="utf-8"))


def _registry():
    from referencing import Registry, Resource

    registry = Registry()
    for path in CONTRACTS.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def integrity_errors(record_type: str, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record_type == "AgentSession":
        started = data.get("started_at")
        last = data.get("last_active_at")
        if started and last and started > last:
            errors.append("started_at must be <= last_active_at")
        if data.get("status") == "closed" and data.get("episode_count", 0) == 0:
            errors.append("a closed session should have at least one episode")

    elif record_type == "AgentEpisode":
        content = data.get("content", "")
        content_hash = data.get("content_hash", "")
        if content and content_hash.startswith("sha256:"):
            import hashlib

            digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest != content_hash:
                errors.append("content_hash must be sha256 of content")
        if data.get("turn_role") == "tool" and not data.get("tool_calls"):
            errors.append("tool turns should reference at least one tool call")

    elif record_type == "ToolResult":
        result = data.get("result", {})
        result_hash = data.get("result_hash", "")
        if result and result_hash.startswith("sha256:"):
            import hashlib

            digest = "sha256:" + hashlib.sha256(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if digest != result_hash:
                errors.append("result_hash must be sha256 of canonical JSON of result")
        started = data.get("started_at")
        completed = data.get("completed_at")
        if started and completed and started > completed:
            errors.append("started_at must be <= completed_at")

    elif record_type == "WebSource":
        if not data.get("trust_label"):
            errors.append("trust_label is required for every web source")

    elif record_type == "UserTopicRecord":
        # mastery_is_supported_by_assessment_evidence: any non-unknown mastery
        # state must trace to an assessment.
        status = data.get("mastery_status")
        assessment = data.get("last_assessment_id")
        if status in ("understood", "applied", "needs_review", "misconception") and not assessment:
            errors.append(f"mastery_status '{status}' requires last_assessment_id (assessment evidence)")
        if status == "unknown" and assessment:
            errors.append("mastery_status 'unknown' must not reference an assessment")
        if data.get("created_at") and data.get("updated_at") and data["created_at"] > data["updated_at"]:
            errors.append("created_at must be <= updated_at")

    elif record_type == "UserEvidence":
        # user_evidence_is_append_only is enforced by the store; here we check
        # that assessment evidence carries its assessment reference.
        if data.get("evidence_type") == "assessment" and not data.get("assessment_id"):
            errors.append("assessment evidence requires assessment_id")
        observed = data.get("observed_at")
        recorded = data.get("recorded_at")
        if observed and recorded and observed > recorded:
            errors.append("observed_at must be <= recorded_at")

    elif record_type == "TopicCluster":
        # topic_clusters_are_derived_not_authoritative: every cluster must
        # trace to at least one source document.
        if not data.get("member_document_ids"):
            errors.append("topic cluster requires at least one member document")
        if not data.get("representative_chunk_id"):
            errors.append("topic cluster requires a representative chunk reference")
        # A cluster cannot be its own parent.
        if data.get("parent_cluster_id") == data.get("cluster_id"):
            errors.append("a topic cluster cannot be its own parent")

    return errors


def validate(record_type: str, data: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator, FormatChecker

    schema = load_schema(record_type)
    validator = Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())
    errors = []
    for error in sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path)
        errors.append(f"{path}: {error.message}" if path else error.message)
    if not errors:
        errors.extend(integrity_errors(record_type, data))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", choices=SCHEMAS)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate(args.record, data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Valid {args.record}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
