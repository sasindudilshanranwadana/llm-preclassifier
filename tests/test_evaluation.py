import json
from pathlib import Path

import pytest

from llm_preclassifier.evaluation import EvalCase, evaluate, load_dataset, main

DATASET = Path(__file__).resolve().parents[1] / "eval" / "dataset.jsonl"


def case(prompt: str, **expected) -> EvalCase:
    return EvalCase(id=prompt, messages=[{"role": "user", "content": prompt}], expected=expected)


def test_load_dataset_accepts_prompt_shorthand(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "Summarise this.", "expected": {"task_type": "summarization"}}\n'
        "\n"
        '{"id": "b", "messages": [{"role": "user", "content": "hi"}], "expected": {"action": "route"}}\n'
    )

    cases = load_dataset(path)

    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].messages == [{"role": "user", "content": "Summarise this."}]


def test_load_dataset_rejects_unknown_expected_fields(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "prompt": "x", "expected": {"mood": "happy"}}\n')

    with pytest.raises(ValueError, match="line 1"):
        load_dataset(path)


def test_evaluate_reports_accuracy_per_field_and_class():
    report = evaluate([
        case("Summarise this report.", task_type="summarization"),
        case("Extract the total from this receipt.", task_type="extraction"),
        case("Extract the total from this receipt.", task_type="summarization"),
    ])

    assert report.total == 3
    assert report.field_accuracy["task_type"] == pytest.approx(2 / 3)
    assert report.per_class["task_type"]["summarization"]["recall"] == pytest.approx(0.5)
    assert report.confusion["task_type"]["summarization"]["extraction"] == 1
    assert [miss.id for miss in report.misses] == ["Extract the total from this receipt."]


def test_evaluate_buckets_confidence_for_calibration():
    report = evaluate([
        case("Extract the invoice number from this email.", task_type="extraction"),
        case("Extract the invoice number from this email.", task_type="coding"),
    ])

    bucket = report.calibration["0.8-0.9"]
    assert bucket["count"] == 2
    assert bucket["accuracy"] == pytest.approx(0.5)
    assert bucket["mean_confidence"] == pytest.approx(0.88)


def test_cli_fails_below_threshold(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "prompt": "Summarise this.", "expected": {"task_type": "coding"}}\n')

    assert main([str(path), "--min-accuracy", "0.5"]) == 1
    assert "task_type" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "prompt": "Summarise this.", "expected": {"task_type": "summarization"}}\n')

    assert main([str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["field_accuracy"]["task_type"] == 1.0


def test_bundled_dataset_meets_ci_floor():
    report = evaluate(load_dataset(DATASET))

    assert report.total >= 40
    assert report.overall_accuracy >= 0.8
    assert report.field_accuracy["action"] == 1.0
