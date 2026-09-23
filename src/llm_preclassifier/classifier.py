"""Deterministic, offline-first request classification."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from llm_preclassifier.catalog import ModelCatalog
from llm_preclassifier.policy import Policy, default_policy
from llm_preclassifier.schemas import ClassificationDecision, Message, ModelRecommendation
from llm_preclassifier.utils import _flatten_text, _has_tool_history

if TYPE_CHECKING:
    from llm_preclassifier.semantic import SemanticClassifier

def _latest_user_text(messages: Iterable[Message]) -> str:
    for message in reversed(list(messages)):
        if message.role == "user":
            return _flatten_text(message.content).strip()
    return ""


def classify(
    messages: list[Message | dict],
    metadata: dict | None = None,
    semantic: SemanticClassifier | None = None,
    policy: Policy | None = None,
    catalog: ModelCatalog | None = None,
) -> ClassificationDecision:
    """Return an explainable decision without sending prompt content anywhere."""
    policy = policy or default_policy()
    patterns = policy.patterns
    confidences = policy.confidence
    normalized = [message if isinstance(message, Message) else Message.model_validate(message) for message in messages]
    metadata = metadata or {}
    latest = _latest_user_text(normalized)
    has_tools = bool(metadata.get("available_tools")) or _has_tool_history([item.model_dump() for item in normalized])

    if not latest or patterns["ambiguous"].match(latest):
        return _decision(policy, catalog, "unknown", "unknown", "unknown", "unknown", 0.0, "unknown",
                         ["insufficient_request_context"])
    verdict = semantic.assess(latest) if semantic else None
    escalation_reasons = _escalation_reasons(latest, metadata, verdict, policy)
    if escalation_reasons:
        return _decision(
            policy, catalog, "unknown", "unknown", "unknown", "human_or_policy_review", 0.5,
            "escalate", escalation_reasons,
        )

    external_action = bool(patterns["external_action"].search(latest))
    requires_tools = external_action or bool(patterns["workspace_action"].search(latest))
    reasons: list[str] = ["offline_rules_v1"]
    task_type = _rule_task_type(latest, external_action, requires_tools, policy)
    mixed = _competing_categories(latest, policy) > 1
    greeting = task_type == "chat" and bool(patterns["greeting"].match(latest))

    use_semantic = (
        verdict is not None and verdict.task_type is not None and not greeting
        and (task_type in {"chat", "classification"}
             or (mixed and verdict.task_share >= policy.semantic.override_share))
    )
    if use_semantic:
        task_type = verdict.task_type
        requires_tools = requires_tools or task_type == "agent_action"
        reasons.append("semantic_task_vote")
    tool_requirement = "required" if requires_tools else ("optional" if has_tools else "none")
    if requires_tools:
        reasons.append("tool_required")

    is_complex = bool(patterns["multi_step"].search(latest)) or (task_type == "coding" and requires_tools)
    complexity = "complex" if is_complex else "simple"
    if is_complex:
        reasons.append("multi_step_or_artifact_signal")

    if use_semantic:
        confidence = round(confidences["semantic_base"] + confidences["semantic_scale"] * verdict.task_share, 2)
    elif greeting:
        confidence = confidences["greeting"]
    elif task_type in {"chat", "classification"}:
        confidence = confidences["fallback"]
        reasons.append("no_category_signal")
    else:
        confidence = confidences["rule_match"]
    if mixed and not use_semantic:
        confidence = min(confidence, confidences["mixed_cap"])
        reasons.append("mixed_signals")
    return _decision(policy, catalog, task_type, complexity, tool_requirement,
                     _tier(task_type, requires_tools, is_complex), min(confidence, 1.0), "route", reasons)


def _escalation_reasons(latest: str, metadata: dict, verdict, policy: Policy) -> list[str]:
    reasons = []
    if policy.patterns["high_stakes"].search(latest) or "high_stakes" in metadata.get("policy_flags", []):
        reasons.append("high_stakes_signal")
    if verdict is not None and verdict.risk_category:
        reasons.append(f"semantic_high_stakes:{verdict.risk_category}")
    return reasons


def _rule_task_type(latest: str, external_action: bool, requires_tools: bool, policy: Policy) -> str:
    coding = bool(policy.patterns["coding"].search(latest))
    if external_action or (requires_tools and not coding):
        return "agent_action"
    if coding:
        return "coding"
    for task_type in policy.category_order:
        if policy.patterns[task_type].search(latest):
            return task_type
    return "chat" if len(latest.split()) <= policy.chat_max_words else "classification"


def _tier(task_type: str, requires_tools: bool, is_complex: bool) -> str:
    if task_type in {"coding", "agent_action"} and requires_tools:
        return "capable"
    if task_type in {"reasoning", "planning", "research"} and is_complex:
        return "reasoning"
    if task_type in {"extraction", "summarization", "classification", "chat"}:
        return "economy"
    return "standard"


def _competing_categories(text: str, policy: Policy) -> int:
    return sum(1 for name in policy.mixed_signal_categories if policy.patterns[name].search(text))


def _decision(
    policy: Policy, catalog: ModelCatalog | None, task_type, complexity, tool_requirement, tier, confidence,
    action, reasons,
) -> ClassificationDecision:
    recommendations = [
        ModelRecommendation(
            provider=rec.provider,
            model=rec.model,
            input_cost_per_million=rec.input_cost_per_million,
            output_cost_per_million=rec.output_cost_per_million,
            currency=catalog.currency,
            notes=rec.notes,
        )
        for rec in (catalog.recommendations_for(tier) if catalog else ())
    ]
    return ClassificationDecision(
        task_type=task_type,
        complexity=complexity,
        tool_requirement=tool_requirement,
        recommended_model_tier=tier,
        confidence=confidence,
        action=action,
        reasons=reasons,
        policy_version=policy.version,
        model_recommendations=recommendations,
    )
