from pathlib import Path

import pytest
import yaml

from llm_preclassifier.classifier import classify
from llm_preclassifier.policy import PolicyError, default_policy, load_policy, main

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "src" / "llm_preclassifier" / "data" / "policy.yaml"


def write_policy(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_PATH.read_text())
    mutate(raw)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def decide(prompt, policy=None):
    return classify([{"role": "user", "content": prompt}], policy=policy)


def test_default_policy_is_versioned_and_hashed():
    policy = default_policy()

    assert policy.version
    assert len(policy.sha256) == 64


def test_decisions_carry_the_policy_version(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw.update(version="acme-2026.10"))

    assert decide("Summarise this report.", load_policy(path)).policy_version == "acme-2026.10"
    assert decide("Summarise this report.").policy_version == default_policy().version


def test_custom_patterns_change_classification(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw["patterns"]["coding"].append("grimoire"))

    assert decide("Tidy up my grimoire.").task_type != "coding"
    assert decide("Tidy up my grimoire.", load_policy(path)).task_type == "coding"


def test_custom_high_stakes_terms_escalate(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw["patterns"]["high_stakes"].append("visa (?:appeal|refusal)"))

    assert decide("Help me write my visa appeal.", load_policy(path)).action == "escalate"


def test_confidence_values_come_from_policy(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw["confidence"].update(rule_match=0.77))

    assert decide("Summarise this report.", load_policy(path)).confidence == 0.77


def test_invalid_regex_names_the_offending_entry(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw["patterns"]["coding"].append("unclosed(group"))

    with pytest.raises(PolicyError, match=r"patterns\.coding\[\d+\]"):
        load_policy(path)


@pytest.mark.parametrize("mutate, message", [
    (lambda raw: raw.update(surprise=True), "unknown keys"),
    (lambda raw: raw["patterns"].pop("writing"), "missing keys"),
    (lambda raw: raw["confidence"].update(fallback=1.5), "between 0 and 1"),
    (lambda raw: raw.update(version="has spaces!"), "version"),
    (lambda raw: raw.update(category_order=["writing", "gossip"]), "category_order"),
    (lambda raw: raw["patterns"].update(coding=[]), "at least one"),
    (lambda raw: raw["learned"].update(enabled="yes"), "learned.enabled"),
    (lambda raw: raw["learned"].update(min_probability=2), "learned.min_probability"),
    (lambda raw: raw["learned"].pop("model"), "missing keys: learned.model"),
    (lambda raw: raw["learned"].update(model="nope.bin"), "learned.model file not found"),
])
def test_invalid_policies_are_rejected(tmp_path, mutate, message):
    with pytest.raises(PolicyError, match=message):
        load_policy(write_policy(tmp_path, mutate))


def test_empty_or_missing_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    with pytest.raises(PolicyError):
        load_policy(empty)
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "absent.yaml")


def test_cli_validates_a_policy(tmp_path, capsys):
    good = write_policy(tmp_path, lambda raw: raw.update(version="ok-1"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: [")

    assert main([str(good)]) == 0
    assert "ok-1" in capsys.readouterr().out
    assert main([str(bad)]) == 1


def test_semantic_exemplars_path_is_relative_to_the_policy_file(tmp_path):
    (tmp_path / "bank.json").write_text('{"chat": ["hello there"]}')
    path = write_policy(tmp_path, lambda raw: raw["semantic"].update(exemplars="bank.json", k=3))

    policy = load_policy(path)

    assert policy.semantic.exemplars == (tmp_path / "bank.json").resolve()
    assert policy.semantic.k == 3


def test_missing_exemplars_file_is_rejected(tmp_path):
    path = write_policy(tmp_path, lambda raw: raw["semantic"].update(exemplars="nope.json"))

    with pytest.raises(PolicyError, match="exemplars"):
        load_policy(path)


def test_policies_without_a_learned_section_use_the_default(tmp_path):
    policy = load_policy(write_policy(tmp_path, lambda raw: raw.pop("learned")))

    assert policy.learned == default_policy().learned
    assert policy.learned.enabled


def test_learned_model_can_be_disabled(tmp_path):
    policy = load_policy(write_policy(tmp_path, lambda raw: raw["learned"].update(enabled=False)))

    result = decide("Is a whale a fish or a mammal?", policy)

    assert result.task_type == "chat"
    assert "learned_task_model" not in result.reasons


def test_learned_threshold_comes_from_policy(tmp_path):
    policy = load_policy(write_policy(tmp_path, lambda raw: raw["learned"].update(min_probability=1.0)))

    assert "learned_task_model" not in decide("Is a whale a fish or a mammal?", policy).reasons


def test_learned_model_path_is_relative_to_the_policy_file(tmp_path):
    (tmp_path / "model.bin").write_bytes(DEFAULT_PATH.with_name("task_model.bin").read_bytes())
    path = write_policy(tmp_path, lambda raw: raw["learned"].update(model="model.bin"))

    policy = load_policy(path)

    assert policy.learned.model == (tmp_path / "model.bin").resolve()
    assert decide("Is a whale a fish or a mammal?", policy).task_type == "classification"
