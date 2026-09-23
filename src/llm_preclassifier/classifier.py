"""Deterministic, offline-first request classification."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

from llm_preclassifier.schemas import ClassificationDecision, Message
from llm_preclassifier.utils import _flatten_text, _has_tool_history

if TYPE_CHECKING:
    from llm_preclassifier.semantic import SemanticClassifier

_HIGH_STAKES = re.compile(
    r"\b(?:diagnos(?:e|is)|medications?|dos(?:e|es|age|ing)|overdos(?:e|ing)|prescriptions?|medical|"
    r"suicid(?:e|al)|self[- ]harm|(?:hurt|harm|kill) (?:myself|themselves|himself|herself)|"
    r"legal advice|lawsuits?|contract dispute|tax return|financial advice|invest(?:ment|ments|ing)?)\b",
    re.IGNORECASE,
)
# Actions on systems outside the workspace; these win over coding vocabulary.
_EXTERNAL_ACTION = re.compile(
    r"\b(?:search (?:the )?web|browse|look up|"
    r"(?:create|open|file|raise) (?:an? |the )?(?:issue|ticket|pull request|pr)|"
    r"send (?:an? |the )?(?:email|message|slack|request|invite)|"
    r"(?:schedule|book) (?:an? )?(?:meeting|call|appointment)|upload|download)\b",
    re.IGNORECASE,
)
# Actions inside the workspace; combined with coding vocabulary they stay coding.
_WORKSPACE_ACTION = re.compile(
    r"\b(?:read (?:the )?(?:file|repository|repo)|run (?:the )?(?:tests?|test suite|command)|"
    r"execute|deploy|create (?:an? )?file|edit (?:the )?(?:file|config))\b",
    re.IGNORECASE,
)
_CODING = re.compile(
    r"\b(?:code|coding|bugs?|(?:unit|failing|integration) tests?|test suite|repository|repo|"
    r"function|class(?:es)? (?:method|definition)|api|python|javascript|typescript|bash|"
    r"powershell|golang|go service|sql|c\+\+|c program|docker(?:file)?|regex|"
    r"segfault|implement|refactor|debug|stack trace|compile)\b",
    re.IGNORECASE,
)
_EXTRACTION = re.compile(
    r"\b(?:extract|parse|pull out|grab|identify|find the|list (?:all|the|every)(?: \w+)? "
    r"(?:names?|dates?|emails?|items?|numbers?|people|companies))\b",
    re.IGNORECASE,
)
_SUMMARIZATION = re.compile(
    r"\b(?:summari[sz](?:e|ing)|tl;?dr|shorten|condense|boil (?:this|it)?(?: \w+){0,2} down|sum up|"
    r"key points|gist)\b",
    re.IGNORECASE,
)
_WRITING = re.compile(
    r"\b(?:write|draft|rewrite|compose|proofread|edit (?:this|my) (?:text|essay|letter)|"
    r"poem|haiku|essay|story|lyrics|cover letter)\b",
    re.IGNORECASE,
)
_RESEARCH = re.compile(r"\b(?:research|compare|latest|current|sources?|citations?|cite)\b", re.IGNORECASE)
_PLANNING = re.compile(
    r"\b(?:plan|project plan|roadmaps?|milestones?|itinerary|strateg(?:y|ies)|architecture|"
    r"organi[sz]e (?:my|our|the))\b",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\b(?:analy[sz]e|reason|explain|evaluate|why|trade-?offs?|pros and cons)\b", re.IGNORECASE,
)
_CHAT = re.compile(
    r"^(?:hi|hey|hello|yo|thanks|thank you|cheers|good (?:morning|afternoon|evening)|"
    r"how are you|what'?s up)\b",
    re.IGNORECASE,
)
_MULTI_STEP = re.compile(r"\b(?:then|after that|and then|end[- ]to[- ]end|multiple|several|all of)\b", re.IGNORECASE)
_AMBIGUOUS = re.compile(r"^(?:help me(?: with this)?|fix it|do it|please help)[.!?\s]*$", re.IGNORECASE)


def _latest_user_text(messages: Iterable[Message]) -> str:
    for message in reversed(list(messages)):
        if message.role == "user":
            return _flatten_text(message.content).strip()
    return ""


def classify(
    messages: list[Message | dict],
    metadata: dict | None = None,
    semantic: SemanticClassifier | None = None,
) -> ClassificationDecision:
    """Return an explainable decision without sending prompt content anywhere."""
    normalized = [message if isinstance(message, Message) else Message.model_validate(message) for message in messages]
    metadata = metadata or {}
    latest = _latest_user_text(normalized)
    has_tools = bool(metadata.get("available_tools")) or _has_tool_history([item.model_dump() for item in normalized])

    if not latest or _AMBIGUOUS.match(latest):
        return _decision("unknown", "unknown", "unknown", "unknown", 0.0, "unknown", ["insufficient_request_context"])
    verdict = semantic.assess(latest) if semantic else None
    escalation_reasons = _escalation_reasons(latest, metadata, verdict)
    if escalation_reasons:
        return _decision(
            "unknown", "unknown", "unknown", "human_or_policy_review", 0.5,
            "escalate", escalation_reasons,
        )

    external_action = bool(_EXTERNAL_ACTION.search(latest))
    requires_tools = external_action or bool(_WORKSPACE_ACTION.search(latest))
    reasons: list[str] = ["offline_rules_v1"]
    task_type = _rule_task_type(latest, external_action, requires_tools)
    mixed = _competing_categories(latest) > 1
    greeting = task_type == "chat" and bool(_CHAT.match(latest))

    use_semantic = (
        verdict is not None and verdict.task_type is not None and not greeting
        and (task_type in {"chat", "classification"} or (mixed and verdict.task_share >= _SEMANTIC_OVERRIDE_SHARE))
    )
    if use_semantic:
        task_type = verdict.task_type
        requires_tools = requires_tools or task_type == "agent_action"
        reasons.append("semantic_task_vote")
    tool_requirement = "required" if requires_tools else ("optional" if has_tools else "none")
    if requires_tools:
        reasons.append("tool_required")

    is_complex = bool(_MULTI_STEP.search(latest)) or (task_type == "coding" and requires_tools)
    complexity = "complex" if is_complex else "simple"
    if is_complex:
        reasons.append("multi_step_or_artifact_signal")

    if use_semantic:
        confidence = round(0.5 + 0.4 * verdict.task_share, 2)
    elif greeting:
        confidence = 0.85
    elif task_type in {"chat", "classification"}:
        confidence = _FALLBACK_CONFIDENCE
        reasons.append("no_category_signal")
    else:
        confidence = 0.88
    if mixed and not use_semantic:
        confidence = min(confidence, 0.7)
        reasons.append("mixed_signals")
    return _decision(task_type, complexity, tool_requirement, _tier(task_type, requires_tools, is_complex),
                     confidence, "route", reasons)


def _escalation_reasons(latest: str, metadata: dict, verdict) -> list[str]:
    reasons = []
    if _HIGH_STAKES.search(latest) or "high_stakes" in metadata.get("policy_flags", []):
        reasons.append("high_stakes_signal")
    if verdict is not None and verdict.risk_category:
        reasons.append(f"semantic_high_stakes:{verdict.risk_category}")
    return reasons


def _rule_task_type(latest: str, external_action: bool, requires_tools: bool) -> str:
    coding = bool(_CODING.search(latest))
    if external_action or (requires_tools and not coding):
        return "agent_action"
    if coding:
        return "coding"
    for task_type, pattern in _ORDERED_CATEGORIES:
        if pattern.search(latest):
            return task_type
    return "chat" if len(latest.split()) <= 12 else "classification"


def _tier(task_type: str, requires_tools: bool, is_complex: bool) -> str:
    if task_type in {"coding", "agent_action"} and requires_tools:
        return "capable"
    if task_type in {"reasoning", "planning", "research"} and is_complex:
        return "reasoning"
    if task_type in {"extraction", "summarization", "classification", "chat"}:
        return "economy"
    return "standard"


_SEMANTIC_OVERRIDE_SHARE = 0.6
_ORDERED_CATEGORIES = (
    ("extraction", _EXTRACTION),
    ("summarization", _SUMMARIZATION),
    ("research", _RESEARCH),
    ("planning", _PLANNING),
    ("writing", _WRITING),
    ("reasoning", _REASONING),
)
_FALLBACK_CONFIDENCE = 0.5
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
