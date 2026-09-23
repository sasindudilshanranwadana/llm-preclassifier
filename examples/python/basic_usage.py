"""Minimal example of using the bundled client SDK against a running instance.

    pip install llm-preclassifier
    uvicorn --factory llm_preclassifier.api:create_app &
    python examples/python/basic_usage.py
"""
from llm_preclassifier.client import PreclassifierClient, PreclassifierError


def main() -> None:
    with PreclassifierClient("http://127.0.0.1:8802") as client:
        print(client.health())

        decision = client.classify(
            messages=[{"role": "user", "content": "Summarise this quarterly report."}],
            available_tools=["terminal"],
        )
        print(f"task_type={decision.task_type} tier={decision.recommended_model_tier} action={decision.action}")
        for recommendation in decision.model_recommendations:
            print(f"  -> {recommendation.provider}/{recommendation.model} ({recommendation.currency})")

        try:
            client.feedback(decision.decision_id, outcome="correct")
        except PreclassifierError as error:
            # /v1/feedback is disabled unless the server sets ENABLE_FEEDBACK=true.
            print(f"feedback not recorded: {error}")


if __name__ == "__main__":
    main()
