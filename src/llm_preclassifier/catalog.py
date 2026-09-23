"""Versioned, operator-editable model catalog: concrete model suggestions per tier.

``recommended_model_tier`` on a decision is deliberately abstract (economy, standard,
capable, reasoning). This module maps each actionable tier to concrete provider/model
picks with illustrative per-million-token costs, so operators can keep pricing current
without a code release. See ``data/model_catalog.yaml`` for the bundled defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

ACTIONABLE_TIERS = ("economy", "standard", "capable", "reasoning")
TOP_LEVEL_KEYS = ("version", "currency", "tiers")
RECOMMENDATION_KEYS = ("provider", "model", "input_cost_per_million", "output_cost_per_million", "notes")
REQUIRED_RECOMMENDATION_KEYS = frozenset(RECOMMENDATION_KEYS) - {"notes"}
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ModelCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class ModelRecommendation:
    provider: str
    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    notes: str | None = None


@dataclass(frozen=True)
class ModelCatalog:
    version: str
    sha256: str
    currency: str
    tiers: Mapping[str, tuple[ModelRecommendation, ...]]

    def recommendations_for(self, tier: str) -> tuple[ModelRecommendation, ...]:
        return self.tiers.get(tier, ())


@lru_cache(maxsize=1)
def default_catalog() -> ModelCatalog:
    source = resources.files("llm_preclassifier").joinpath("data/model_catalog.yaml")
    return parse_catalog(source.read_text(encoding="utf-8"), "bundled model catalog")


def load_catalog(path: str | Path | None = None) -> ModelCatalog:
    if not path:
        return default_catalog()
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ModelCatalogError(f"model catalog {path}: cannot read file: {error.strerror}") from error
    return parse_catalog(text, f"model catalog {path}")


def parse_catalog(text: str, label: str) -> ModelCatalog:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ModelCatalogError(f"{label}: invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ModelCatalogError(f"{label}: expected a mapping at the top level")

    def fail(message: str) -> ModelCatalogError:
        return ModelCatalogError(f"{label}: {message}")

    _check_keys(raw, TOP_LEVEL_KEYS, "", fail)
    version = str(raw["version"])
    if not _VERSION.match(version):
        raise fail("version must be 1-64 characters of letters, digits, '.', '_' or '-'")
    currency = raw["currency"]
    if not isinstance(currency, str) or not currency.strip():
        raise fail("currency must be a non-empty string")

    tiers_raw = raw["tiers"]
    _check_keys(tiers_raw, ACTIONABLE_TIERS, "tiers.", fail)
    tiers = {
        tier: tuple(
            _recommendation(tier, index, entry, fail) for index, entry in enumerate(_entries(tier, tiers_raw, fail))
        )
        for tier in ACTIONABLE_TIERS
    }

    return ModelCatalog(
        version=version,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        currency=currency,
        tiers=MappingProxyType(tiers),
    )


def _entries(tier: str, tiers_raw: dict, fail) -> list:
    entries = tiers_raw[tier]
    if not isinstance(entries, list) or not entries:
        raise fail(f"tiers.{tier} must be a list with at least one recommendation")
    return entries


def _recommendation(tier: str, index: int, entry, fail) -> ModelRecommendation:
    prefix = f"tiers.{tier}[{index}]"
    if not isinstance(entry, dict):
        raise fail(f"{prefix} must be a mapping")
    _check_keys(entry, RECOMMENDATION_KEYS, f"{prefix}.", fail, required=REQUIRED_RECOMMENDATION_KEYS)
    provider = _nonempty_str(entry["provider"], f"{prefix}.provider", fail)
    model = _nonempty_str(entry["model"], f"{prefix}.model", fail)
    input_cost = _non_negative(entry["input_cost_per_million"], f"{prefix}.input_cost_per_million", fail)
    output_cost = _non_negative(entry["output_cost_per_million"], f"{prefix}.output_cost_per_million", fail)
    notes = entry.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise fail(f"{prefix}.notes must be a string")
    return ModelRecommendation(provider, model, input_cost, output_cost, notes)


def _check_keys(section, expected, prefix: str, fail, required=None) -> None:
    if not isinstance(section, dict):
        raise fail(f"{prefix.rstrip('.') or 'catalog'} must be a mapping")
    required_keys = set(required) if required is not None else set(expected)
    unknown = sorted(set(section) - set(expected))
    missing = sorted(required_keys - set(section))
    if unknown:
        raise fail(f"unknown keys: {', '.join(prefix + key for key in unknown)}")
    if missing:
        raise fail(f"missing keys: {', '.join(prefix + key for key in missing)}")


def _nonempty_str(value, key: str, fail) -> str:
    if not isinstance(value, str) or not value.strip():
        raise fail(f"{key} must be a non-empty string")
    return value


def _non_negative(value, key: str, fail) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise fail(f"{key} must be a non-negative number")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a model catalog file.")
    parser.add_argument("path", nargs="?", help="catalog file; omit to check the bundled default")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.path)
    except ModelCatalogError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(f"OK version={catalog.version} sha256={catalog.sha256}")
    return 0
