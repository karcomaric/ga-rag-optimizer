"""Tests for ``src.evaluation.loader``."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.loader import load_qa_split, load_sentence_document, load_sentences
from src.schemas import QASample, SupportingFact

SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "qa_pairs_split.json"


def test_load_qa_split_returns_train_and_test() -> None:
    split = load_qa_split(SPLIT_PATH)

    assert set(split.keys()) == {"train", "test"}
    assert len(split["train"]) == 100
    assert len(split["test"]) == 50


def test_all_entries_are_qasample_dataclasses() -> None:
    split = load_qa_split(SPLIT_PATH)

    for entry in split["train"] + split["test"]:
        assert isinstance(entry, QASample)
        assert isinstance(entry.id, str) and entry.id
        assert isinstance(entry.question, str) and entry.question
        assert isinstance(entry.answer, str)
        assert entry.type in {"bridge", "comparison"}


def test_supporting_facts_are_dataclasses_with_int_sent_id() -> None:
    split = load_qa_split(SPLIT_PATH)

    for entry in split["train"]:
        assert entry.supporting_facts, f"QA {entry.id} hat keine supporting_facts"
        for fact in entry.supporting_facts:
            assert isinstance(fact, SupportingFact)
            assert isinstance(fact.paragraph_id, str) and fact.paragraph_id
            assert isinstance(fact.sent_id, int)
            assert fact.sent_id >= 0


def test_load_sentences_skips_title_and_blank_lines(tmp_path: Path) -> None:
    corpus_file = tmp_path / "sample_paragraph.txt"
    content = "Sample Title\n\nFirst sentence.\nSecond sentence.\n\n"
    corpus_file.write_text(content, encoding="utf-8")

    sentences = load_sentences(tmp_path, "sample_paragraph")

    assert sentences == ["First sentence.", "Second sentence."]


def test_load_sentence_document_wraps_sentence_with_metadata(tmp_path: Path) -> None:
    corpus_file = tmp_path / "sample_paragraph.txt"
    content = "Sample Title\n\nFirst sentence.\nSecond sentence.\n"
    corpus_file.write_text(content, encoding="utf-8")

    doc = load_sentence_document(tmp_path, "sample_paragraph", sent_id=1)

    assert doc.page_content == "Second sentence."
    assert doc.metadata == {"paragraph_id": "sample_paragraph"}


def test_train_and_test_do_not_overlap() -> None:
    split = load_qa_split(SPLIT_PATH)
    train_ids = {qa.id for qa in split["train"]}
    test_ids = {qa.id for qa in split["test"]}

    assert not (train_ids & test_ids), "Train und Test teilen IDs"
