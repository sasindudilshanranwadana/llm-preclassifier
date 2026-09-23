import hashlib
import re

import pytest

np = pytest.importorskip("numpy")

from llm_preclassifier.classifier import classify
from llm_preclassifier.semantic import SemanticClassifier, load_exemplars

DIMENSIONS = 256


class BagOfWordsEmbedder:
    """Deterministic stand-in: texts sharing words get similar vectors."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(DIMENSIONS)
            for word in re.findall(r"[a-z]+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1
            vectors.append(vector)
        return vectors


BANK = {
    "risk:medical": ["my chest hurts badly", "pain in my chest and arm"],
    "coding": ["fix this python function", "python function throws error"],
    "writing": ["write a poem about rain", "write a short poem"],
}


@pytest.fixture
def semantic():
    return SemanticClassifier(BANK, BagOfWordsEmbedder(), k=2, risk_share=0.5, min_similarity=0.3)


def test_nearest_risk_exemplars_trigger_escalation_category(semantic):
    verdict = semantic.assess("my chest hurts")

    assert verdict.risk_category == "medical"


def test_task_vote_ignores_risk_labels(semantic):
    verdict = semantic.assess("please write a poem")

    assert verdict.risk_category is None
    assert verdict.task_type == "writing"
    assert 0.0 < verdict.task_share <= 1.0


def test_unrelated_text_yields_no_verdict(semantic):
    verdict = semantic.assess("zebra quantum marmalade")

    assert verdict.risk_category is None
    assert verdict.task_type is None


def test_classifier_escalates_on_semantic_risk_the_rules_miss(semantic):
    result = classify([{"role": "user", "content": "my chest hurts"}], semantic=semantic)

    assert result.action == "escalate"
    assert "semantic_high_stakes:medical" in result.reasons


def test_classifier_uses_semantic_task_when_rules_have_no_signal(semantic):
    result = classify([{"role": "user", "content": "something about rain"}], semantic=semantic)

    assert result.task_type == "writing"
    assert "semantic_task_vote" in result.reasons


def test_rules_win_when_they_have_a_clear_signal(semantic):
    result = classify([{"role": "user", "content": "Summarise this report."}], semantic=semantic)

    assert result.task_type == "summarization"
    assert "semantic_task_vote" not in result.reasons


def test_semantic_agent_action_requires_tools():
    bank = {"agent_action": ["post update to the team channel", "post notes to channel"]}
    semantic = SemanticClassifier(bank, BagOfWordsEmbedder(), k=2, risk_share=0.5, min_similarity=0.3)

    result = classify([{"role": "user", "content": "post the notes to our channel"}], semantic=semantic)

    assert result.task_type == "agent_action"
    assert result.tool_requirement == "required"


def test_bundled_exemplars_cover_every_task_and_risk_label():
    bank = load_exemplars()

    assert {"risk:medical", "risk:legal", "risk:financial", "risk:self_harm"} <= set(bank)
    assert all(len(texts) >= 5 for texts in bank.values())


def test_unknown_bank_label_is_rejected():
    with pytest.raises(ValueError, match="unknown exemplar label"):
        SemanticClassifier({"gossip": ["hi"]}, BagOfWordsEmbedder())
