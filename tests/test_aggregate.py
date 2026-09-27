"""Tests for ``src.evaluation.aggregate``."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.evaluation.aggregate import (
    evaluate_questions,
    mean_retrieval_metrics,
    print_metrics_table,
)
from src.schemas import QASample, SupportingFact


def _sample(sample_id: str, question: str, answer: str, paragraph_id: str) -> QASample:
    return QASample(
        id=sample_id,
        question=question,
        answer=answer,
        type="bridge",
        level="easy",
        supporting_facts=[SupportingFact(paragraph_id=paragraph_id, sent_id=0)],
        context_paragraph_ids=[paragraph_id],
    )


def test_mean_over_hit_and_miss() -> None:
    hit = _sample("q1", "Who founded Standard Oil?", "Standard Oil", "p1")
    miss = _sample("q2", "Where is the Eiffel Tower?", "Paris", "p2")
    responses = {
        hit.question: [
            Document(page_content="Standard Oil was founded.", metadata={"paragraph_id": "p1"})
        ],
        miss.question: [
            Document(page_content="Completely unrelated text.", metadata={"paragraph_id": "p9"})
        ],
    }
    retriever = RunnableLambda(lambda query: responses[query])

    result = mean_retrieval_metrics(retriever, [hit, miss])

    assert result["context_recall"] == pytest.approx(0.5)
    assert result["context_precision"] == pytest.approx(0.5)
    assert result["answer_f1"] == pytest.approx(0.5)
    assert result["exact_match"] == pytest.approx(0.5)


def test_perfect_retrieval_gives_ones() -> None:
    qa = _sample("q1", "Who founded Standard Oil?", "Standard Oil", "p1")
    retriever = RunnableLambda(
        lambda query: [
            Document(page_content="Standard Oil was founded.", metadata={"paragraph_id": "p1"})
        ]
    )

    result = mean_retrieval_metrics(retriever, [qa])

    assert result == pytest.approx(
        {"context_recall": 1.0, "context_precision": 1.0, "answer_f1": 1.0, "exact_match": 1.0}
    )


def test_evaluate_questions_keeps_per_question_details() -> None:
    qa = _sample("q1", "Who founded Standard Oil?", "Standard Oil", "p1")
    retriever = RunnableLambda(
        lambda query: [
            Document(page_content="Standard Oil was founded.", metadata={"paragraph_id": "p1"})
        ]
    )

    per_question = evaluate_questions(retriever, [qa])

    assert len(per_question) == 1
    assert per_question[0]["id"] == "q1"
    assert per_question[0]["retrieved"] == ["p1"]
    assert per_question[0]["context_recall"] == pytest.approx(1.0)


def test_empty_questions_raises() -> None:
    retriever = RunnableLambda(lambda query: [])

    with pytest.raises(ValueError, match="leer"):
        mean_retrieval_metrics(retriever, [])


def test_print_metrics_table_formats_rows(capsys: pytest.CaptureFixture[str]) -> None:
    results = {
        "default_langchain": {
            "context_recall": 0.5,
            "context_precision": 0.25,
            "answer_f1": 0.125,
            "exact_match": 1.0,
        },
    }

    print_metrics_table(results)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["Konfiguration", "recall", "precision", "answer_f1", "exact_match"]
    assert set(lines[1]) == {"-"}
    assert lines[2].split() == ["default_langchain", "0.500", "0.250", "0.125", "1.000"]
