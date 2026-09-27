"""Dataclass ``QASample``."""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.supporting_fact import SupportingFact


@dataclass(frozen=True)
class QASample:
    """A HotpotQA question with gold answer and supporting evidence.

    ``type`` is ``bridge`` or ``comparison``, ``context_paragraph_ids`` holds
    all 10 paragraphs of the original example, evidence plus distractors.
    """

    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: list[SupportingFact]
    context_paragraph_ids: list[str]
