"""Aggregation of per-question retrieval metrics over a question set."""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from src.evaluation.metrics import answer_f1, context_precision, context_recall, exact_match
from src.schemas import QASample, QuestionEval


def evaluate_questions(
    retriever: Runnable[str, list[Document]],
    questions: list[QASample],
) -> list[QuestionEval]:
    """Evaluates all questions against one retriever and collects the per-question details.

    Raises:
        ValueError: If ``questions`` is empty.
    """
    if not questions:
        raise ValueError("questions darf nicht leer sein")

    per_question: list[QuestionEval] = []
    for qa in questions:
        retrieved = retriever.invoke(qa.question)
        per_question.append(
            QuestionEval(
                id=qa.id,
                context_recall=context_recall(retrieved, qa.supporting_facts),
                context_precision=context_precision(retrieved, qa.supporting_facts),
                answer_f1=answer_f1(retrieved, qa.answer),
                exact_match=exact_match(retrieved, qa.answer),
                retrieved=[chunk.metadata["paragraph_id"] for chunk in retrieved],
            )
        )
    return per_question


def mean_metrics(per_question: list[QuestionEval]) -> dict[str, float]:
    """Averages the four metrics over all evaluated questions."""
    return {
        "context_recall": float(np.mean([q["context_recall"] for q in per_question])),
        "context_precision": float(np.mean([q["context_precision"] for q in per_question])),
        "answer_f1": float(np.mean([q["answer_f1"] for q in per_question])),
        "exact_match": float(np.mean([q["exact_match"] for q in per_question])),
    }


def mean_retrieval_metrics(
    retriever: Runnable[str, list[Document]],
    questions: list[QASample],
) -> dict[str, float]:
    """Computes the mean retrieval metrics for a retriever over a question set.

    Convenience wrapper for callers that do not need the per-question details.

    Raises:
        ValueError: If ``questions`` is empty.
    """
    return mean_metrics(evaluate_questions(retriever, questions))


def print_metrics_table(
    results: dict[str, dict[str, float]],
    row_label: str = "Konfiguration",
) -> None:
    """Prints the retrieval metrics per row as a fixed-width text table."""
    width = max([len(row_label), *(len(name) for name in results)])
    header = (
        f"{row_label:<{width}s} {'recall':>8s} {'precision':>10s} "
        f"{'answer_f1':>10s} {'exact_match':>12s}"
    )
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(
            f"{name:<{width}s} "
            f"{scores['context_recall']:>8.3f} "
            f"{scores['context_precision']:>10.3f} "
            f"{scores['answer_f1']:>10.3f} "
            f"{scores['exact_match']:>12.3f}"
        )
