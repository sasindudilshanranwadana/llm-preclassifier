from llm_preclassifier.classifier import classify


def decision(prompt: str, **metadata):
    return classify([{"role": "user", "content": prompt}], metadata)


def test_simple_extraction_uses_economy_tier():
    result = decision("Extract the invoice number from this email.")

    assert result.task_type == "extraction"
    assert result.complexity == "simple"
    assert result.tool_requirement == "none"
    assert result.recommended_model_tier == "economy"
    assert result.action == "route"


def test_code_change_with_tools_requires_capable_tier():
    result = decision(
        "Inspect this repository, fix the failing tests, then run the test suite.",
        available_tools=["filesystem", "terminal"],
    )

    assert result.task_type == "coding"
    assert result.complexity == "complex"
    assert result.tool_requirement == "required"
    assert result.recommended_model_tier == "capable"
    assert "tool_required" in result.reasons


def test_high_stakes_request_escalates_without_claiming_safety():
    result = decision("Tell me the correct medication dose for my child.")

    assert result.complexity == "unknown"
    assert result.recommended_model_tier == "human_or_policy_review"
    assert result.action == "escalate"
    assert "high_stakes_signal" in result.reasons


def test_ambiguous_prompt_abstains():
    result = decision("Help me with this")

    assert result.task_type == "unknown"
    assert result.complexity == "unknown"
    assert result.action == "unknown"


def test_explicit_agent_action_requires_tools():
    result = decision("Search the web for the latest release notes.")

    assert result.task_type == "agent_action"
    assert result.tool_requirement == "required"
    assert result.recommended_model_tier == "capable"


def test_everyday_words_do_not_trigger_coding():
    assert decision("Help me organise my class schedule for next semester.").task_type != "coding"
    assert decision("Can you explain how a driving test is scored?").task_type != "coding"


def test_send_alone_does_not_require_tools():
    result = decision("Send my regards in this thank-you note draft.")

    assert result.task_type == "writing"
    assert result.tool_requirement == "none"


def test_send_email_is_an_agent_action():
    result = decision("Send an email to the team about the outage.")

    assert result.task_type == "agent_action"
    assert result.tool_requirement == "required"


def test_competing_signals_lower_confidence():
    clear = decision("Extract the invoice number from this email.")
    mixed = decision("Research competitors, write a draft, and summarise the key points.")

    assert mixed.confidence < clear.confidence
    assert "mixed_signals" in mixed.reasons


def test_dosage_and_self_harm_phrasings_escalate():
    for prompt in (
        "What dosage of ibuprofen is safe with my blood thinners?",
        "I want to hurt myself.",
        "How many pills would be an overdose?",
    ):
        assert decision(prompt).action == "escalate", prompt


def test_external_actions_beat_coding_words():
    result = decision("Create an issue for the login bug we discussed.")

    assert result.task_type == "agent_action"
    assert result.tool_requirement == "required"


def test_unrecognised_short_prompt_is_low_confidence_chat():
    greeting = decision("Hello there!")
    unmatched = decision("What is the capital of France?")

    assert greeting.task_type == unmatched.task_type == "chat"
    assert unmatched.confidence < greeting.confidence


def test_learned_model_routes_prompts_the_rules_miss():
    result = decision("Is a whale a fish or a mammal?")

    assert result.task_type == "classification"
    assert "learned_task_model" in result.reasons
    assert "no_category_signal" not in result.reasons
    assert result.confidence >= 0.6


def test_learned_agent_action_requires_tools():
    result = decision("I need to know the weather in Tokyo right now.")

    assert result.task_type == "agent_action"
    assert result.tool_requirement == "required"


def test_clear_rule_matches_are_not_overridden_by_the_learned_model():
    result = decision("Extract the invoice number from this email.")

    assert "learned_task_model" not in result.reasons


def test_model_recommendations_are_populated_from_the_given_catalog():
    from llm_preclassifier.catalog import default_catalog

    decision = classify(
        [{"role": "user", "content": "Summarise this report."}],
        catalog=default_catalog(),
    )

    assert decision.recommended_model_tier == "economy"
    assert decision.model_recommendations
    assert all(rec.currency == "USD" for rec in decision.model_recommendations)


def test_model_recommendations_are_empty_without_a_catalog():
    decision = classify([{"role": "user", "content": "Summarise this report."}])

    assert decision.model_recommendations == []


def test_escalated_decisions_have_no_model_recommendations():
    from llm_preclassifier.catalog import default_catalog

    decision = classify(
        [{"role": "user", "content": "What dosage of ibuprofen is safe?"}],
        catalog=default_catalog(),
    )

    assert decision.action == "escalate"
    assert decision.model_recommendations == []
