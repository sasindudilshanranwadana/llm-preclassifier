import json

from llm_preclassifier.audit import append_decision_log
from llm_preclassifier.schemas import ClassificationDecision


def test_decision_log_never_accepts_or_records_prompt_content(tmp_path):
    path = tmp_path / "decisions.jsonl"
    decision = ClassificationDecision(
        task_type="summarization",
        complexity="simple",
        tool_requirement="none",
        recommended_model_tier="economy",
        confidence=0.88,
        action="route",
        reasons=["offline_rules_v1"],
    )

    append_decision_log(str(path), decision)

    record = json.loads(path.read_text())
    assert "prompt" not in record
    assert "content" not in record
    assert record["task_type"] == "summarization"


def test_feedback_log_never_accepts_or_records_prompt_content(tmp_path):
    from llm_preclassifier.audit import append_feedback_log
    from llm_preclassifier.schemas import FeedbackRequest

    path = tmp_path / "feedback.jsonl"
    feedback = FeedbackRequest(
        decision_id="abc123",
        outcome="incorrect",
        corrected_task_type="coding",
    )

    append_feedback_log(str(path), feedback)

    record = json.loads(path.read_text())
    assert "prompt" not in record
    assert "content" not in record
    assert record["decision_id"] == "abc123"
    assert record["outcome"] == "incorrect"
    assert record["corrected_task_type"] == "coding"
