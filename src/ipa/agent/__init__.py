"""Agent core: identity, durable sessions, episodic memory and deterministic tools.

The agent is omnipresent: one identity and one memory shared by every surface
(CLI, dashboard). Surfaces are thin clients that open sessions; they never
define personality nor keep agent state (DEC-002).
"""
from .agent_identity import Identity, load_identity
from .agent_memory import AgentMemory
from .agent_core import AgentCore
from .agent_tools import (
    TOOL_NAMES,
    CoverageAssessment,
    ToolCall,
    ToolContext,
    ToolResult,
    assess_corpus_coverage,
    execute_tool,
)
from .judge import HeuristicJudge, Judgment, LLMJudge
from .provider_wiring import build_llm_judge, build_responder
from .compile_report_executor import CompileReportResult, execute_compile_report
from .research_executor import SourceJudgment, WebSource, ResearchResult, execute_research
from .web_search import search_web

__all__ = [
    "AgentCore",
    "AgentMemory",
    "Identity",
    "load_identity",
    "TOOL_NAMES",
    "ToolCall",
    "ToolResult",
    "ToolContext",
    "execute_tool",
    "CoverageAssessment",
    "assess_corpus_coverage",
    "HeuristicJudge",
    "Judgment",
    "LLMJudge",
    "build_llm_judge",
    "build_responder",
    "SourceJudgment",
    "WebSource",
    "ResearchResult",
    "execute_research",
    "CompileReportResult",
    "execute_compile_report",
    "search_web",
]
