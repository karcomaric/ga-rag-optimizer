"""TypedDict ``FitnessResult`` and type alias ``FitnessFn``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from src.schemas.question_eval import QuestionEval


class FitnessResult(TypedDict):
    """Return contract of the fitness callable.

    ``composite`` is the weighted score the GA maximizes, the four metric
    fields are the unweighted means over all evaluated questions.
    """

    composite: float
    context_recall: float
    context_precision: float
    answer_f1: float
    exact_match: float
    per_question: list[QuestionEval]


FitnessFn = Callable[[dict[str, Any]], FitnessResult]
