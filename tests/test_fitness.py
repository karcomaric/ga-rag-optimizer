"""Tests for ``src.evaluation.fitness``."""

from __future__ import annotations

import pytest

from src.evaluation.fitness import composite_score


def test_happy_path_default_weights() -> None:
    metrics = {"context_recall": 0.5, "context_precision": 0.4, "answer_f1": 0.6}
    weights = {"context_recall": 0.4, "context_precision": 0.3, "answer_f1": 0.3}

    score = composite_score(metrics, weights)

    assert score == pytest.approx(0.5 * 0.4 + 0.4 * 0.3 + 0.6 * 0.3)


def test_missing_metric_key_raises() -> None:
    metrics = {"context_recall": 0.5, "context_precision": 0.4}
    weights = {"context_recall": 0.4, "context_precision": 0.3, "answer_f1": 0.3}

    with pytest.raises(ValueError, match="answer_f1"):
        composite_score(metrics, weights)


def test_weights_not_summing_to_one_raises() -> None:
    metrics = {"a": 1.0, "b": 1.0}
    weights = {"a": 0.5, "b": 0.4}

    with pytest.raises(ValueError, match="1.0"):
        composite_score(metrics, weights)


def test_all_one_metrics_give_one() -> None:
    metrics = {"context_recall": 1.0, "context_precision": 1.0, "answer_f1": 1.0}
    weights = {"context_recall": 0.4, "context_precision": 0.3, "answer_f1": 0.3}

    assert composite_score(metrics, weights) == pytest.approx(1.0)
