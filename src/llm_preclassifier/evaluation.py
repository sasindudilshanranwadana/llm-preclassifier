"""Offline accuracy evaluation against a labeled JSONL dataset.

Each line is ``{"id", "prompt" | "messages", "available_tools"?, "policy_flags"?, "expected"}``
where ``expected`` holds any subset of the decision fields in ``EVALUATED_FIELDS``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_preclassifier.classifier import classify
from llm_preclassifier.policy import Policy, PolicyError, default_policy, load_policy

EVALUATED_FIELDS = ("task_type", "complexity", "tool_requirement", "recommended_model_tier", "action")


@dataclass(frozen=True)
class EvalCase:
    id: str
    messages: list[dict]
    expected: dict[str, str]
    available_tools: list[str] = field(default_factory=list)
    policy_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Miss:
    id: str
    field: str
    expected: str
    actual: str


@dataclass(frozen=True)
class EvalReport:
    policy_version: str
    total: int
    overall_accuracy: float
    field_accuracy: dict[str, float]
    per_class: dict[str, dict[str, dict[str, float]]]
    confusion: dict[str, dict[str, dict[str, int]]]
    calibration: dict[str, dict[str, float]]
    misses: list[Miss]


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(_parse_case(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: line {line_number}: {error}") from error
    return cases


def _parse_case(raw: dict) -> EvalCase:
    expected = raw["expected"]
    unknown = set(expected) - set(EVALUATED_FIELDS)
    if unknown:
        raise ValueError(f"unknown expected fields {sorted(unknown)}")
    messages = raw.get("messages") or [{"role": "user", "content": raw["prompt"]}]
    return EvalCase(
        id=str(raw["id"]),
        messages=messages,
        expected=expected,
        available_tools=raw.get("available_tools", []),
        policy_flags=raw.get("policy_flags", []),
    )


def evaluate(cases: list[EvalCase], semantic=None, policy: Policy | None = None) -> EvalReport:
    policy = policy or default_policy()
    field_hits: Counter[str] = Counter()
    field_totals: Counter[str] = Counter()
    confusion: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    buckets: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    misses: list[Miss] = []
    fully_correct = 0

    for case in cases:
        decision = classify(case.messages, {
            "available_tools": case.available_tools,
            "policy_flags": case.policy_flags,
        }, semantic=semantic, policy=policy).model_dump()
        case_correct = True
        for name, expected in case.expected.items():
            actual = decision[name]
            field_totals[name] += 1
            confusion[name][expected][actual] += 1
            if actual == expected:
                field_hits[name] += 1
            else:
                case_correct = False
                misses.append(Miss(case.id, name, expected, actual))
        fully_correct += case_correct
        buckets[_bucket(decision["confidence"])].append((decision["confidence"], case_correct))

    return EvalReport(
        policy_version=policy.version,
        total=len(cases),
        overall_accuracy=fully_correct / len(cases) if cases else 0.0,
        field_accuracy={name: field_hits[name] / field_totals[name] for name in field_totals},
        per_class={name: _per_class(matrix) for name, matrix in confusion.items()},
        confusion={name: {label: dict(row) for label, row in matrix.items()} for name, matrix in confusion.items()},
        calibration={label: _calibration(points) for label, points in sorted(buckets.items())},
        misses=misses,
    )


def _bucket(confidence: float) -> str:
    lower = min(int(confidence * 10), 9) / 10
    return f"{lower:.1f}-{lower + 0.1:.1f}"


def _calibration(points: list[tuple[float, bool]]) -> dict[str, float]:
    return {
        "count": len(points),
        "mean_confidence": sum(confidence for confidence, _ in points) / len(points),
        "accuracy": sum(correct for _, correct in points) / len(points),
    }


def _per_class(matrix: dict[str, Counter[str]]) -> dict[str, dict[str, float]]:
    labels = set(matrix) | {actual for row in matrix.values() for actual in row}
    stats = {}
    for label in sorted(labels):
        true_positive = matrix.get(label, Counter())[label]
        support = sum(matrix.get(label, Counter()).values())
        predicted = sum(row[label] for row in matrix.values())
        stats[label] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
        }
    return stats


def format_report(report: EvalReport) -> str:
    lines = [f"policy: {report.policy_version}", f"cases: {report.total}", f"overall (all fields correct): {report.overall_accuracy:.1%}", ""]
    lines += [f"  {name:<24} {accuracy:.1%}" for name, accuracy in report.field_accuracy.items()]
    lines += ["", "task_type per class:"]
    for label, stats in report.per_class.get("task_type", {}).items():
        lines.append(
            f"  {label:<16} support={stats['support']:<4.0f} "
            f"precision={stats['precision']:.2f} recall={stats['recall']:.2f}"
        )
    lines += ["", "calibration:"]
    for label, stats in report.calibration.items():
        lines.append(
            f"  {label}  n={stats['count']:<4.0f} "
            f"mean_conf={stats['mean_confidence']:.2f} accuracy={stats['accuracy']:.2f}"
        )
    if report.misses:
        lines += ["", "misses:"]
        lines += [f"  {m.id}: {m.field} expected={m.expected} actual={m.actual}" for m in report.misses]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the classifier against a labeled JSONL dataset.")
    parser.add_argument("dataset", help="path to a JSONL dataset")
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="fail if overall accuracy is below this")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--semantic", metavar="MODEL", help="enable the semantic layer with this embedding model")
    parser.add_argument("--policy", metavar="PATH", help="policy file to evaluate (default: bundled policy)")
    args = parser.parse_args(argv)

    try:
        policy = load_policy(args.policy)
    except PolicyError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2
    semantic = None
    if args.semantic:
        from llm_preclassifier.semantic import build_semantic_classifier

        semantic = build_semantic_classifier(args.semantic, os.getenv("SEMANTIC_CACHE_DIR"), policy)
    report = evaluate(load_dataset(args.dataset), semantic, policy)
    print(json.dumps(asdict(report), indent=2) if args.json else format_report(report))
    if report.overall_accuracy < args.min_accuracy:
        print(f"\nFAIL: overall accuracy {report.overall_accuracy:.1%} < {args.min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
