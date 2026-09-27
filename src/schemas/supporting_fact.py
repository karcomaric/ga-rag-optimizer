"""Dataclass ``SupportingFact``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportingFact:
    """Reference to a single evidence sentence in the corpus.

    ``paragraph_id`` is the file stem under ``data/corpus/``, ``sent_id`` is
    zero-indexed within that paragraph.
    """

    paragraph_id: str
    sent_id: int
