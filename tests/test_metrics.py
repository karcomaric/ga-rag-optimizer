"""Tests for ``src.evaluation.metrics``."""

from __future__ import annotations

from langchain_core.documents import Document

from src.evaluation.metrics import (
    _normalize_answer,
    _token_f1,
    answer_f1,
    context_precision,
    context_recall,
    exact_match,
)
from src.schemas import SupportingFact


def _doc(paragraph_id: str, text: str = "...") -> Document:
    """Builds a Document with a paragraph_id, as the pipeline delivers it."""
    return Document(page_content=text, metadata={"paragraph_id": paragraph_id})


def test_context_recall_all_paragraphs_in_retrieval() -> None:
    gold = [SupportingFact("p1", 0), SupportingFact("p2", 3)]
    retrieved = [_doc("p1"), _doc("p2")]

    assert context_recall(retrieved, gold) == 1.0


def test_context_recall_nothing_found() -> None:
    gold = [SupportingFact("p1", 0), SupportingFact("p2", 3)]
    retrieved = [_doc("p9")]

    assert context_recall(retrieved, gold) == 0.0


def test_context_recall_partial_match_unique_paragraphs() -> None:
    gold = [
        SupportingFact("p1", 0),
        SupportingFact("p1", 5),
        SupportingFact("p2", 3),
        SupportingFact("p3", 1),
    ]
    retrieved = [_doc("p1"), _doc("p2")]

    assert context_recall(retrieved, gold) == 2 / 3


def test_context_recall_empty_gold_returns_one() -> None:
    retrieved = [_doc("p1")]

    assert context_recall(retrieved, []) == 1.0


def test_context_recall_empty_retrieval_returns_zero_when_gold_present() -> None:
    gold = [SupportingFact("p1", 0)]

    assert context_recall([], gold) == 0.0


def test_context_precision_all_chunks_from_gold_paragraph() -> None:
    gold = [SupportingFact("p1", 0), SupportingFact("p2", 3)]
    retrieved = [_doc("p1"), _doc("p2")]

    assert context_precision(retrieved, gold) == 1.0


def test_context_precision_half_relevant() -> None:
    gold = [SupportingFact("p1", 0)]
    retrieved = [_doc("p1"), _doc("p9")]

    assert context_precision(retrieved, gold) == 0.5


def test_context_precision_empty_retrieval_returns_zero() -> None:
    gold = [SupportingFact("p1", 0)]

    assert context_precision([], gold) == 0.0


def test_context_precision_no_gold_returns_zero() -> None:
    retrieved = [_doc("p1")]

    assert context_precision(retrieved, []) == 0.0


def test_normalize_answer() -> None:
    assert _normalize_answer("The Cat's name!") == "cats name"
    assert _normalize_answer("  A   quick  Brown Fox  ") == "quick brown fox"
    assert _normalize_answer("An apple.") == "apple"
    assert _normalize_answer("") == ""


def test_token_f1_identical_strings_return_one() -> None:
    assert _token_f1("hello world", "hello world") == 1.0


def test_token_f1_order_insensitive() -> None:
    assert _token_f1("world hello", "hello world") == 1.0


def test_token_f1_no_overlap_returns_zero() -> None:
    assert _token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_known_fraction() -> None:
    assert _token_f1("hello world test", "hello world") == 0.8


def test_answer_f1_match_in_one_chunk() -> None:
    retrieved = [
        _doc("p1", "completely irrelevant content here"),
        _doc("p2", "The answer is Marco Karic indeed"),
    ]
    assert answer_f1(retrieved, "Marco Karic") == 1.0


def test_answer_f1_no_match_returns_zero() -> None:
    retrieved = [_doc("p1", "totally unrelated text here")]
    assert answer_f1(retrieved, "Marco Karic") == 0.0


def test_answer_f1_empty_retrieval_returns_zero() -> None:
    assert answer_f1([], "answer") == 0.0


def test_answer_f1_partial_match_returns_known_fraction() -> None:
    retrieved = [_doc("p1", "His name is Marco Lewis.")]
    assert answer_f1(retrieved, "Marco Karic") == 0.5


def test_answer_f1_chunk_shorter_than_answer() -> None:
    retrieved = [_doc("p1", "Marco Karic")]
    assert answer_f1(retrieved, "Marco Karic Wins") == 0.8


def test_exact_match_token_based_no_false_substring_hit() -> None:
    retrieved = [_doc("p1", "Apart from this")]
    assert exact_match(retrieved, "art") == 0.0


def test_exact_match_contiguous_sequence_hits() -> None:
    retrieved = [_doc("p1", "His name is Marco Karic.")]
    assert exact_match(retrieved, "Marco Karic") == 1.0


def test_exact_match_normalization_applies() -> None:
    retrieved = [_doc("p1", "The Cat")]
    assert exact_match(retrieved, "the cat") == 1.0
    assert exact_match(retrieved, "Cat!") == 1.0


def test_exact_match_empty_gold_returns_zero() -> None:
    retrieved = [_doc("p1", "something")]
    assert exact_match(retrieved, "") == 0.0
