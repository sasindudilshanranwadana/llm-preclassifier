"""Safe optional JSONL decision metadata logging."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from llm_preclassifier.schemas import ClassificationDecision, FeedbackRequest


def append_decision_log(path: str, decision: ClassificationDecision) -> None:
    """Write allowlisted decision metadata; request text is never accepted here."""
    record = {
        "time": datetime.now(UTC).isoformat(),
        "decision_id": decision.decision_id,
        "task_type": decision.task_type,
        "complexity": decision.complexity,
        "tool_requirement": decision.tool_requirement,
        "recommended_model_tier": decision.recommended_model_tier,
        "confidence": decision.confidence,
        "action": decision.action,
        "reasons": decision.reasons,
        "policy_version": decision.policy_version,
    }
    _append_jsonl(path, record)


def append_feedback_log(path: str, feedback: FeedbackRequest) -> None:
    """Write allowlisted correction metadata; the schema itself forbids prompt content."""
    record = {
        "time": datetime.now(UTC).isoformat(),
        "decision_id": feedback.decision_id,
        "outcome": feedback.outcome,
        "corrected_task_type": feedback.corrected_task_type,
        "corrected_complexity": feedback.corrected_complexity,
        "corrected_tool_requirement": feedback.corrected_tool_requirement,
        "corrected_recommended_model_tier": feedback.corrected_recommended_model_tier,
    }
    _append_jsonl(path, record)


def _append_jsonl(path: str, record: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
