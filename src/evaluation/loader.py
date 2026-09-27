"""Loader for the HotpotQA evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.schemas import QASample, SupportingFact


def load_qa_split(path: Path) -> dict[str, list[QASample]]:
    """Loads the train/test split from ``qa_pairs_split.json``.

    Returns a dict with the keys ``train`` and ``test``, order matches the file.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[QASample]] = {}
    for split in ("train", "test"):
        result[split] = [_to_sample(entry) for entry in raw[split]]
    return result


def load_sentences(corpus_dir: Path, paragraph_id: str) -> list[str]:
    """Reads the sentences of a corpus paragraph, indexable by ``sent_id``.

    The file holds the title in line 1, a blank line, then one sentence per
    line. ``paragraph_id`` is the file stem, without the txt suffix.
    """
    lines = (corpus_dir / f"{paragraph_id}.txt").read_text(encoding="utf-8").splitlines()
    return [line for line in lines[2:] if line.strip()]


def load_sentence_document(
    corpus_dir: Path,
    paragraph_id: str,
    sent_id: int = 0,
) -> Document:
    """Wraps a single corpus sentence in a Document with its ``paragraph_id``."""
    text = load_sentences(corpus_dir, paragraph_id)[sent_id]
    return Document(page_content=text, metadata={"paragraph_id": paragraph_id})


def _to_sample(entry: dict[str, Any]) -> QASample:
    """Converts a JSON entry into a ``QASample`` with typed ``SupportingFact`` items."""
    return QASample(
        id=entry["id"],
        question=entry["question"],
        answer=entry["answer"],
        type=entry["type"],
        level=entry.get("level", ""),
        supporting_facts=[
            SupportingFact(paragraph_id=f["paragraph_id"], sent_id=int(f["sent_id"]))
            for f in entry["supporting_facts"]
        ],
        context_paragraph_ids=list(entry["context_paragraph_ids"]),
    )
