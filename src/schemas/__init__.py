"""Shared data structures used by all modules of the GA-RAG optimizer."""

from __future__ import annotations

from src.schemas.fitness_result import FitnessFn, FitnessResult
from src.schemas.ga_result import GAResult
from src.schemas.optimizer_result import OptimizerResult
from src.schemas.qa_sample import QASample
from src.schemas.question_eval import QuestionEval
from src.schemas.supporting_fact import SupportingFact

__all__ = [
    "FitnessFn",
    "FitnessResult",
    "GAResult",
    "OptimizerResult",
    "QASample",
    "QuestionEval",
    "SupportingFact",
]
