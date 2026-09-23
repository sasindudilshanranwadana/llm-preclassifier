"""Optional offline semantic layer: nearest-exemplar voting over local embeddings.

Install with ``pip install llm-preclassifier[semantic]``. Embeddings run on CPU
through ONNX (fastembed); request text never leaves the process.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol, Sequence, get_args

import numpy as np

from llm_preclassifier.policy import Policy, default_policy
from llm_preclassifier.schemas import TaskType

RISK_PREFIX = "risk:"
RISK_CATEGORIES = frozenset({"medical", "legal", "financial", "self_harm"})
_TASK_LABELS = frozenset(get_args(TaskType)) - {"unknown"}


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class SemanticVerdict:
    risk_category: str | None
    task_type: str | None
    task_share: float


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=cache_dir or None)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return list(self._model.embed(list(texts)))


def load_exemplars(path: Path | None = None) -> dict[str, list[str]]:
    source = path or resources.files("llm_preclassifier").joinpath("data/exemplars.json")
    return json.loads(source.read_text(encoding="utf-8"))


def build_semantic_classifier(
    model_name: str, cache_dir: str | None = None, policy: Policy | None = None,
) -> SemanticClassifier:
    settings = (policy or default_policy()).semantic
    return SemanticClassifier(
        load_exemplars(settings.exemplars),
        FastEmbedEmbedder(model_name, cache_dir),
        k=settings.k,
        risk_share=settings.risk_share,
        min_similarity=settings.min_similarity,
    )


class SemanticClassifier:
    """Vote among the k most similar labeled exemplars.

    Risk is flagged when risk labels hold at least ``risk_share`` of the vote
    weight, deliberately lower than a majority so near-misses escalate.
    """

    def __init__(
        self,
        bank: dict[str, list[str]],
        embedder: Embedder,
        k: int = 7,
        risk_share: float = 0.35,
        min_similarity: float = 0.55,
    ) -> None:
        for label in bank:
            if label not in _TASK_LABELS and label.removeprefix(RISK_PREFIX) not in RISK_CATEGORIES:
                raise ValueError(f"unknown exemplar label: {label}")
        self._embedder = embedder
        self._k = k
        self._risk_share = risk_share
        self._min_similarity = min_similarity
        self._labels = [label for label, texts in bank.items() for _ in texts]
        texts = [text for texts in bank.values() for text in texts]
        self._matrix = _normalize(np.asarray(embedder.embed(texts), dtype=np.float32))

    def assess(self, text: str) -> SemanticVerdict:
        query = _normalize(np.asarray(self._embedder.embed([text]), dtype=np.float32))[0]
        similarities = self._matrix @ query
        nearest = [i for i in np.argsort(similarities)[::-1][: self._k] if similarities[i] >= self._min_similarity]
        if not nearest:
            return SemanticVerdict(None, None, 0.0)

        weights: dict[str, float] = defaultdict(float)
        for index in nearest:
            weights[self._labels[index]] += float(similarities[index])
        total = sum(weights.values())

        risk_weights = {label: weight for label, weight in weights.items() if label.startswith(RISK_PREFIX)}
        risk_category = None
        if sum(risk_weights.values()) / total >= self._risk_share:
            risk_category = max(risk_weights, key=risk_weights.get).removeprefix(RISK_PREFIX)

        task_weights = {label: weight for label, weight in weights.items() if label in _TASK_LABELS}
        if not task_weights:
            return SemanticVerdict(risk_category, None, 0.0)
        task_type = max(task_weights, key=task_weights.get)
        return SemanticVerdict(risk_category, task_type, task_weights[task_type] / sum(task_weights.values()))


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
