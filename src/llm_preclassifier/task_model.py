"""Learned task-type model: hashed n-gram logistic regression with int8 weights.

Inference is pure Python so the default rules-only install gains it without new
dependencies or network access. Features must stay byte-identical between
training (``training/train_task_model.py``) and inference, so both use
``features`` from this module.
"""
from __future__ import annotations

import json
import math
import re
import struct
import zlib
from array import array
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from itertools import pairwise
from pathlib import Path

MAGIC = b"LPTM1\n"
DEFERRED_LABEL = "other"
MAX_CHARS = 2000
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


class TaskModelError(ValueError):
    pass


def features(text: str, buckets: int) -> dict[int, float]:
    """Unigram, bigram and first-token features hashed into ``buckets``, L2-normalized."""
    tokens = _TOKEN.findall(text[:MAX_CHARS].lower())
    names = {f"w:{token}" for token in tokens}
    names.update(f"b:{left} {right}" for left, right in pairwise(tokens))
    if tokens:
        names.add(f"f:{tokens[0]}")
    indices: dict[int, float] = {}
    for name in names:
        index = zlib.crc32(name.encode("utf-8")) % buckets
        indices[index] = indices.get(index, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in indices.values())) or 1.0
    return {index: value / norm for index, value in indices.items()}


@dataclass(frozen=True)
class TaskPrediction:
    task_type: str
    probability: float


class TaskModel:
    def __init__(self, labels: list[str], buckets: int, scales: list[float], bias: list[float],
                 weights: array, metadata: dict) -> None:
        if len(weights) != len(labels) * buckets or len(scales) != len(labels) or len(bias) != len(labels):
            raise TaskModelError("weight shape does not match labels and buckets")
        self.labels = tuple(labels)
        self.buckets = buckets
        self.metadata = metadata
        self._scales = scales
        self._bias = bias
        self._weights = weights

    @classmethod
    def from_bytes(cls, blob: bytes) -> TaskModel:
        if not blob.startswith(MAGIC):
            raise TaskModelError("not a task model file")
        offset = len(MAGIC)
        try:
            (header_length,) = struct.unpack_from("<I", blob, offset)
            header = json.loads(blob[offset + 4: offset + 4 + header_length])
            weights = array("b", blob[offset + 4 + header_length:])
            return cls(header["labels"], header["buckets"], header["scales"], header["bias"], weights,
                       header.get("metadata", {}))
        except (struct.error, KeyError, TypeError, ValueError) as error:
            raise TaskModelError(f"corrupt task model: {error}") from error

    def to_bytes(self) -> bytes:
        header = json.dumps({
            "labels": list(self.labels), "buckets": self.buckets, "scales": self._scales,
            "bias": self._bias, "metadata": self.metadata,
        }, sort_keys=True).encode("utf-8")
        return MAGIC + struct.pack("<I", len(header)) + header + self._weights.tobytes()

    def predict(self, text: str) -> TaskPrediction:
        scores = list(self._bias)
        for index, value in features(text, self.buckets).items():
            for label_index in range(len(self.labels)):
                weight = self._weights[label_index * self.buckets + index]
                scores[label_index] += weight * self._scales[label_index] * value
        top = max(scores)
        exps = [math.exp(score - top) for score in scores]
        best = exps.index(max(exps))
        return TaskPrediction(self.labels[best], exps[best] / sum(exps))


def load_task_model(path: str | Path | None = None) -> TaskModel:
    if path is None:
        return bundled_task_model()
    try:
        return TaskModel.from_bytes(Path(path).read_bytes())
    except OSError as error:
        raise TaskModelError(f"task model {path}: cannot read file: {error.strerror}") from error


@lru_cache(maxsize=1)
def bundled_task_model() -> TaskModel:
    return TaskModel.from_bytes(
        resources.files("llm_preclassifier").joinpath("data/task_model.bin").read_bytes()
    )
