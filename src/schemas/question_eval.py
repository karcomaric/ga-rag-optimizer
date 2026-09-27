"""TypedDict ``QuestionEval`` for per-question evaluation details."""

from __future__ import annotations

from typing import TypedDict


class QuestionEval(TypedDict):
    """Per-question detail of a single fitness evaluation.

    ``retrieved`` holds the paragraph IDs of the retrieved chunks in rank order.
    """

    id: str
    context_recall: float
    context_precision: float
    answer_f1: float
    exact_match: float
    retrieved: list[str]
