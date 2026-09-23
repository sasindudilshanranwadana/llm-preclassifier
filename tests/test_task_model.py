from array import array

import pytest

from llm_preclassifier.task_model import (
    DEFERRED_LABEL,
    MAGIC,
    TaskModel,
    TaskModelError,
    bundled_task_model,
    features,
    load_task_model,
)


def tiny_model():
    labels = ["coding", DEFERRED_LABEL]
    buckets = 8
    weights = array("b", [0] * len(labels) * buckets)
    for index in features("python", buckets):
        weights[index] = 127
    return TaskModel(labels, buckets, [0.1, 0.1], [0.0, 0.0], weights, {"note": "test"})


def test_features_are_deterministic_and_normalized():
    first = features("Write a Python function", 1 << 15)

    assert first == features("write a python FUNCTION", 1 << 15)
    assert abs(sum(value * value for value in first.values()) - 1.0) < 1e-9
    assert features("", 1 << 15) == {}


def test_features_ignore_text_beyond_the_character_limit():
    prefix = "summarise this " * 200

    assert features(prefix + "python", 64) == features(prefix + "javascript", 64)


def test_model_roundtrips_through_bytes():
    model = tiny_model()

    restored = TaskModel.from_bytes(model.to_bytes())

    assert restored.labels == model.labels
    assert restored.metadata == {"note": "test"}
    assert restored.predict("python") == model.predict("python")
    assert restored.predict("python").task_type == "coding"


@pytest.mark.parametrize("blob, message", [
    (b"not a model", "not a task model"),
    (MAGIC + b"\x01", "corrupt"),
    (MAGIC + b"\x05\x00\x00\x00{bad}", "corrupt"),
])
def test_corrupt_files_are_rejected(blob, message):
    with pytest.raises(TaskModelError, match=message):
        TaskModel.from_bytes(blob)


def test_weight_shape_is_checked():
    blob = tiny_model().to_bytes()

    with pytest.raises(TaskModelError, match="shape"):
        TaskModel.from_bytes(blob[:-1])


def test_missing_model_file_is_reported(tmp_path):
    with pytest.raises(TaskModelError, match="cannot read file"):
        load_task_model(tmp_path / "missing.bin")


def test_bundled_model_loads_with_provenance():
    model = bundled_task_model()

    assert DEFERRED_LABEL in model.labels
    assert {"agent_action", "coding", "writing"} <= set(model.labels)
    assert model.metadata["sources"]["dolly"]["revision"]
    assert load_task_model() is model
