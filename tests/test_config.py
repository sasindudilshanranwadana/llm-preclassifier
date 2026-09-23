from llm_preclassifier.config import Settings


def test_settings_read_environment_at_construction(monkeypatch):
    monkeypatch.setenv("MAX_MESSAGES", "7")

    assert Settings().max_messages == 7


def test_importing_api_does_not_build_an_app(monkeypatch):
    import importlib

    import llm_preclassifier.api as api

    monkeypatch.setenv("LLM_PRECLASSIFIER_ENV", "production")
    monkeypatch.setenv("CLIENT_API_KEYS", "")

    importlib.reload(api)

    assert not hasattr(api, "app")
