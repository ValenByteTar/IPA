"""Configuration for the domain-agnostic Reporter Agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReporterPeriodConfig:
    start: str
    end: str
    label: str


@dataclass(frozen=True)
class ReporterConfig:
    corpus_id: str = "default"
    period: ReporterPeriodConfig = field(default_factory=lambda: ReporterPeriodConfig("", "", "unspecified"))
    min_documents: int = 2
    allow_singleton_topics: bool = True
    max_hierarchy_depth: int = 2
    similarity_threshold: float = 0.52
    weights: dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.25, "source_quality": 0.20, "source_diversity": 0.20,
        "potential_impact": 0.20, "user_interest": 0.15,
    })
    interests: tuple[str, ...] = ()
    quality_threshold: float = 0.25
    research_enabled: bool = True
    research_require_approval: bool = True
    research_budget: dict[str, int] = field(default_factory=lambda: {
        "max_urls": 20, "max_seconds": 600, "max_bytes": 50_000_000, "max_depth": 2,
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReporterConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = data.get("reporter", data)
        period = raw.get("period", {})
        category = raw.get("category_generation", {})
        ranking = raw.get("ranking", {})
        research = raw.get("research", {})
        return cls(
            corpus_id=str(raw.get("corpus_id", "default")),
            period=ReporterPeriodConfig(
                start=str(period.get("start", "")), end=str(period.get("end", "")),
                label=str(period.get("label", "unspecified")),
            ),
            min_documents=int(category.get("min_documents", 2)),
            allow_singleton_topics=bool(category.get("allow_singleton_topics", True)),
            max_hierarchy_depth=int(category.get("max_hierarchy_depth", 2)),
            similarity_threshold=float(category.get("similarity_threshold", 0.52)),
            weights={str(k): float(v) for k, v in ranking.items()} or cls().weights,
            interests=tuple(str(item) for item in raw.get("interests", [])),
            quality_threshold=float(raw.get("quality_threshold", 0.25)),
            research_enabled=bool(research.get("enabled", True)),
            research_require_approval=bool(research.get("require_approval", True)),
            research_budget={
                "max_urls": int(research.get("max_urls", 20)),
                "max_seconds": int(research.get("max_seconds", 600)),
                "max_bytes": int(research.get("max_bytes", 50_000_000)),
                "max_depth": int(research.get("max_depth", 2)),
            },
        )

    def validate(self) -> None:
        if not self.corpus_id or not self.period.label or not self.period.start or not self.period.end:
            raise ValueError("Reporter corpus_id, period label, start and end are required")
        if self.min_documents < 1 or not 0 < self.similarity_threshold <= 1:
            raise ValueError("invalid Reporter clustering configuration")
        if self.max_hierarchy_depth < 1:
            raise ValueError("max_hierarchy_depth must be positive")
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError("Reporter ranking weights must sum to 1")
        for key, value in self.research_budget.items():
            if value < 0:
                raise ValueError(f"research budget {key} must be non-negative")

