"""Safe optional JSONL decision metadata logging."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from llm_preclassifier.schemas import ClassificationDecision


def append_decision_log(path: str, decision: ClassificationDecision) -> None:
    """Write allowlisted decision metadata; request text is never accepted here."""
    destination = Path(path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "time": datetime.now(UTC).isoformat(),
        "task_type": decision.task_type,
        "complexity": decision.complexity,
        "tool_requirement": decision.tool_requirement,
        "recommended_model_tier": decision.recommended_model_tier,
        "confidence": decision.confidence,
        "action": decision.action,
        "reasons": decision.reasons,
        "policy_version": decision.policy_version,
    }
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
