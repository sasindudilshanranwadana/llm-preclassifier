"""Versioned routing policy: patterns, category order, confidence and semantic thresholds.

The bundled ``data/policy.yaml`` is the default. Operators can supply their own file
through ``POLICY_PATH``; it is validated at startup so a bad policy fails fast.
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

SEARCH_PATTERNS = (
    "high_stakes", "external_action", "workspace_action", "coding", "extraction",
    "summarization", "writing", "research", "planning", "reasoning", "multi_step",
)
PATTERN_KEYS = (*SEARCH_PATTERNS, "greeting", "ambiguous")
ORDERABLE_CATEGORIES = frozenset({"extraction", "summarization", "research", "planning", "writing", "reasoning"})
MIXABLE_CATEGORIES = ORDERABLE_CATEGORIES | {"coding"}
CONFIDENCE_KEYS = ("rule_match", "greeting", "fallback", "mixed_cap", "semantic_base", "semantic_scale")
SEMANTIC_KEYS = ("k", "risk_share", "min_similarity", "override_share", "exemplars")
TOP_LEVEL_KEYS = (
    "version", "patterns", "category_order", "mixed_signal_categories",
    "chat_max_words", "confidence", "semantic",
)
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticSettings:
    k: int
    risk_share: float
    min_similarity: float
    override_share: float
    exemplars: Path | None


@dataclass(frozen=True)
class Policy:
    version: str
    sha256: str
    patterns: Mapping[str, re.Pattern[str]]
    category_order: tuple[str, ...]
    mixed_signal_categories: tuple[str, ...]
    chat_max_words: int
    confidence: Mapping[str, float]
    semantic: SemanticSettings


@lru_cache(maxsize=1)
def default_policy() -> Policy:
    source = resources.files("llm_preclassifier").joinpath("data/policy.yaml")
    return parse_policy(source.read_text(encoding="utf-8"), "bundled policy", base_dir=None)


def load_policy(path: str | Path | None = None) -> Policy:
    if not path:
        return default_policy()
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PolicyError(f"policy {path}: cannot read file: {error.strerror}") from error
    return parse_policy(text, f"policy {path}", base_dir=path.parent)


def parse_policy(text: str, label: str, base_dir: Path | None) -> Policy:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PolicyError(f"{label}: invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise PolicyError(f"{label}: expected a mapping at the top level")

    def fail(message: str) -> PolicyError:
        return PolicyError(f"{label}: {message}")

    _check_keys(raw, TOP_LEVEL_KEYS, "", fail)
    version = str(raw["version"])
    if not _VERSION.match(version):
        raise fail("version must be 1-64 characters of letters, digits, '.', '_' or '-'")

    _check_keys(raw["patterns"], PATTERN_KEYS, "patterns.", fail)
    patterns = {key: _compile(key, raw["patterns"][key], fail) for key in PATTERN_KEYS}

    category_order = _categories(raw["category_order"], ORDERABLE_CATEGORIES, "category_order", fail)
    mixed = _categories(raw["mixed_signal_categories"], MIXABLE_CATEGORIES, "mixed_signal_categories", fail)

    chat_max_words = raw["chat_max_words"]
    if not isinstance(chat_max_words, int) or chat_max_words < 1:
        raise fail("chat_max_words must be a positive integer")

    _check_keys(raw["confidence"], CONFIDENCE_KEYS, "confidence.", fail)
    confidence = {key: _unit(raw["confidence"][key], f"confidence.{key}", fail) for key in CONFIDENCE_KEYS}

    return Policy(
        version=version,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        patterns=MappingProxyType(patterns),
        category_order=category_order,
        mixed_signal_categories=mixed,
        chat_max_words=chat_max_words,
        confidence=MappingProxyType(confidence),
        semantic=_semantic(raw["semantic"], base_dir, fail),
    )


def _check_keys(section, expected: tuple[str, ...], prefix: str, fail) -> None:
    if not isinstance(section, dict):
        raise fail(f"{prefix.rstrip('.') or 'policy'} must be a mapping")
    unknown = sorted(set(section) - set(expected))
    missing = sorted(set(expected) - set(section))
    if unknown:
        raise fail(f"unknown keys: {', '.join(prefix + key for key in unknown)}")
    if missing:
        raise fail(f"missing keys: {', '.join(prefix + key for key in missing)}")


def _compile(key: str, alternatives, fail) -> re.Pattern[str]:
    if not isinstance(alternatives, list) or not alternatives:
        raise fail(f"patterns.{key} must be a list with at least one entry")
    for index, alternative in enumerate(alternatives):
        if not isinstance(alternative, str) or not alternative.strip():
            raise fail(f"patterns.{key}[{index}] must be a non-empty string")
        try:
            re.compile(alternative)
        except re.error as error:
            raise fail(f"patterns.{key}[{index}]: invalid regex {alternative!r}: {error}") from error
    body = "|".join(alternatives)
    if key == "greeting":
        source = rf"^(?:{body})\b"
    elif key == "ambiguous":
        source = rf"^(?:{body})[.!?\s]*$"
    else:
        source = rf"\b(?:{body})\b"
    return re.compile(source, re.IGNORECASE)


def _categories(values, allowed: frozenset[str], key: str, fail) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise fail(f"{key} must be a non-empty list")
    unknown = sorted(set(values) - allowed)
    if unknown or len(set(values)) != len(values):
        raise fail(f"{key} must list distinct values from {sorted(allowed)}; got {values}")
    return tuple(values)


def _unit(value, key: str, fail) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise fail(f"{key} must be a number between 0 and 1")
    return float(value)


def _semantic(section, base_dir: Path | None, fail) -> SemanticSettings:
    _check_keys(section, SEMANTIC_KEYS, "semantic.", fail)
    k = section["k"]
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise fail("semantic.k must be a positive integer")
    exemplars = section["exemplars"]
    if exemplars is not None:
        if base_dir is None:
            raise fail("semantic.exemplars must be null in the bundled policy")
        exemplars = (base_dir / str(exemplars)).resolve()
        if not exemplars.is_file():
            raise fail(f"semantic.exemplars file not found: {exemplars}")
    return SemanticSettings(
        k=k,
        risk_share=_unit(section["risk_share"], "semantic.risk_share", fail),
        min_similarity=_unit(section["min_similarity"], "semantic.min_similarity", fail),
        override_share=_unit(section["override_share"], "semantic.override_share", fail),
        exemplars=exemplars,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a routing policy file.")
    parser.add_argument("path", nargs="?", help="policy file; omit to check the bundled default")
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.path)
    except PolicyError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(f"OK version={policy.version} sha256={policy.sha256}")
    return 0

