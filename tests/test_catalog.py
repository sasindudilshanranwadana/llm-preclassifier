from pathlib import Path

import pytest
import yaml

from llm_preclassifier.catalog import ModelCatalogError, default_catalog, load_catalog, main

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "src" / "llm_preclassifier" / "data" / "model_catalog.yaml"


def write_catalog(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_PATH.read_text())
    mutate(raw)
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_catalog_is_versioned_and_hashed():
    catalog = default_catalog()

    assert catalog.version
    assert len(catalog.sha256) == 64
    assert catalog.currency == "USD"


def test_every_actionable_tier_has_at_least_one_recommendation():
    catalog = default_catalog()

    for tier in ("economy", "standard", "capable", "reasoning"):
        recommendations = catalog.recommendations_for(tier)
        assert recommendations
        assert all(rec.provider and rec.model for rec in recommendations)


def test_unknown_tiers_have_no_recommendations():
    catalog = default_catalog()

    assert catalog.recommendations_for("human_or_policy_review") == ()
    assert catalog.recommendations_for("unknown") == ()


def test_custom_catalog_changes_recommendations(tmp_path):
    path = write_catalog(tmp_path, lambda raw: raw["tiers"].update(economy=[
        {"provider": "acme", "model": "acme-nano", "input_cost_per_million": 0.1, "output_cost_per_million": 0.2},
    ]))

    catalog = load_catalog(path)

    assert [rec.model for rec in catalog.recommendations_for("economy")] == ["acme-nano"]


@pytest.mark.parametrize("mutate, message", [
    (lambda raw: raw.update(surprise=True), "unknown keys"),
    (lambda raw: raw["tiers"].pop("reasoning"), "missing keys"),
    (lambda raw: raw["tiers"].update(economy=[]), "at least one recommendation"),
    (lambda raw: raw.update(version="has spaces!"), "version"),
    (lambda raw: raw["tiers"]["economy"][0].pop("model"), "missing keys"),
    (lambda raw: raw["tiers"]["economy"][0].update(input_cost_per_million=-1), "non-negative"),
    (lambda raw: raw.update(currency=""), "currency"),
])
def test_invalid_catalogs_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ModelCatalogError, match=message):
        load_catalog(write_catalog(tmp_path, mutate))


def test_empty_or_missing_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    with pytest.raises(ModelCatalogError):
        load_catalog(empty)
    with pytest.raises(ModelCatalogError, match="cannot read"):
        load_catalog(tmp_path / "absent.yaml")


def test_cli_validates_a_catalog(tmp_path, capsys):
    good = write_catalog(tmp_path, lambda raw: raw.update(version="ok-1"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: [")

    assert main([str(good)]) == 0
    assert "ok-1" in capsys.readouterr().out
    assert main([str(bad)]) == 1
