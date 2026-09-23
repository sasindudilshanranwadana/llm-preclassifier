"""Deterministic, offline-first request classification."""
from __future__ import annotations

import re
from typing import Iterable

from llm_preclassifier.schemas import ClassificationDecision, Message
from llm_preclassifier.utils import _flatten_text, _has_tool_history

_HIGH_STAKES = re.compile(
    r"\b(?:diagnos(?:e|is)|medication|dose|prescription|medical|suicide|self-harm|"
    r"legal advice|lawsuit|contract dispute|tax return|financial advice|invest(?:ment|ing))\b",
    re.IGNORECASE,
)
_TOOL_ACTION = re.compile(
    r"\b(?:search (?:the )?web|browse|look up|read (?:the )?(?:file|repository|repo)|"
    r"run (?:the )?(?:test|command)|execute|deploy|create (?:a )?(?:file|issue|pull request)|"
    r"edit (?:the )?(?:file|config)|send (?:an? |the )?(?:email|message|slack|request|invite)|upload|download)\b",
    re.IGNORECASE,
)
_CODING = re.compile(
    r"\b(?:code|coding|bugs?|(?:unit|failing|integration) tests?|test suite|repository|repo|"
    r"function|class(?:es)? (?:method|definition)|api|python|javascript|typescript|docker|"
    r"implement|refactor|debug|stack trace|compile)\b",
    re.IGNORECASE,
)
_EXTRACTION = re.compile(r"\b(?:extract|parse|pull out|find the)\b", re.IGNORECASE)
_SUMMARIZATION = re.compile(r"\b(?:summari[sz]e|tldr|shorten|key points)\b", re.IGNORECASE)
_WRITING = re.compile(r"\b(?:write|draft|rewrite|proofread|edit this text)\b", re.IGNORECASE)
_RESEARCH = re.compile(r"\b(?:research|compare|latest|current|source|citation)\b", re.IGNORECASE)
_PLANNING = re.compile(r"\b(?:plan|roadmap|milestone|strategy|architecture)\b", re.IGNORECASE)
_REASONING = re.compile(r"\b(?:analyse|analyze|reason|explain|evaluate|why|trade-?off)\b", re.IGNORECASE)
_MULTI_STEP = re.compile(r"\b(?:then|after that|and then|end[- ]to[- ]end|multiple|several|all of)\b", re.IGNORECASE)
_AMBIGUOUS = re.compile(r"^(?:help me(?: with this)?|fix it|do it|please help)[.!?\s]*$", re.IGNORECASE)


def _latest_user_text(messages: Iterable[Message]) -> str:
    for message in reversed(list(messages)):
        if message.role == "user":
            return _flatten_text(message.content).strip()
    return ""


def classify(messages: list[Message | dict], metadata: dict | None = None) -> ClassificationDecision:
    """Return an explainable decision without sending prompt content anywhere."""
    normalized = [message if isinstance(message, Message) else Message.model_validate(message) for message in messages]
    metadata = metadata or {}
    latest = _latest_user_text(normalized)
    has_tools = bool(metadata.get("available_tools")) or _has_tool_history([item.model_dump() for item in normalized])

    if not latest or _AMBIGUOUS.match(latest):
        return _decision("unknown", "unknown", "unknown", "unknown", 0.0, "unknown", ["insufficient_request_context"])
    if _HIGH_STAKES.search(latest) or "high_stakes" in metadata.get("policy_flags", []):
        return _decision(
            "unknown", "unknown", "unknown", "human_or_policy_review", 0.5,
            "escalate", ["high_stakes_signal"],
        )

    requires_tools = bool(_TOOL_ACTION.search(latest))
    tool_requirement = "required" if requires_tools else ("optional" if has_tools else "none")
    reasons: list[str] = ["offline_rules_v1"]
    if requires_tools:
        reasons.append("tool_required")

    if _TOOL_ACTION.search(latest) and not _CODING.search(latest):
        task_type = "agent_action"
    elif _CODING.search(latest):
        task_type = "coding"
    elif _EXTRACTION.search(latest):
        task_type = "extraction"
    elif _SUMMARIZATION.search(latest):
        task_type = "summarization"
    elif _RESEARCH.search(latest):
        task_type = "research"
    elif _PLANNING.search(latest):
        task_type = "planning"
    elif _WRITING.search(latest):
        task_type = "writing"
    elif _TOOL_ACTION.search(latest):
        task_type = "agent_action"
    elif _REASONING.search(latest):
        task_type = "reasoning"
    elif len(latest.split()) <= 12:
        task_type = "chat"
    else:
        task_type = "classification"

    is_complex = bool(_MULTI_STEP.search(latest)) or (task_type == "coding" and requires_tools)
    complexity = "complex" if is_complex else "simple"
    if is_complex:
        reasons.append("multi_step_or_artifact_signal")

    if task_type in {"coding", "agent_action"} and requires_tools:
        tier = "capable"
    elif task_type in {"reasoning", "planning", "research"} and is_complex:
        tier = "reasoning"
    elif task_type in {"extraction", "summarization", "classification", "chat"}:
        tier = "economy"
    else:
        tier = "standard"

    confidence = 0.88 if task_type not in {"chat", "classification"} else 0.65
    if _competing_categories(latest) > 1:
        confidence = min(confidence, 0.7)
        reasons.append("mixed_signals")
    return _decision(task_type, complexity, tool_requirement, tier, confidence, "route", reasons)


_CATEGORY_PATTERNS = (_CODING, _EXTRACTION, _SUMMARIZATION, _WRITING, _RESEARCH, _PLANNING)


def _competing_categories(text: str) -> int:
    return sum(1 for pattern in _CATEGORY_PATTERNS if pattern.search(text))


def _decision(task_type, complexity, tool_requirement, tier, confidence, action, reasons) -> ClassificationDecision:
    return ClassificationDecision(
        task_type=task_type,
        complexity=complexity,
        tool_requirement=tool_requirement,
        recommended_model_tier=tier,
        confidence=confidence,
        action=action,
        reasons=reasons,
    )
