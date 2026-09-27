"""Reference-based metrics for the GA fitness.

All functions work LLM-free on the retrieved chunks and the gold supporting
facts, which keeps the evaluation deterministic. Chunks are
``langchain_core.documents.Document``, the pipeline sets the ``paragraph_id``
key in ``metadata`` when splitting.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from langchain_core.documents import Document

from src.schemas import SupportingFact

_ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_answer(text: str) -> str:
    """Normalizes a string.

    Lowercase, articles ``a``, ``an``, ``the`` removed, punctuation stripped,
    whitespace collapsed to single spaces.
    """
    text = text.lower()
    text = _ARTICLES_PATTERN.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _best_span_f1(chunk_tokens: list[str], gold_norm: str, span_len: int) -> float:
    """Best token F1 of any window of length ``span_len`` in ``chunk_tokens``.

    A chunk shorter than the gold answer forms a single window, the whole chunk.
    """
    if not chunk_tokens:
        return 0.0
    n_windows = max(len(chunk_tokens) - span_len + 1, 1)
    best = 0.0
    for start in range(n_windows):
        window = chunk_tokens[start : start + span_len]
        best = max(best, _token_f1(" ".join(window), gold_norm))
    return best


def _contains_span(tokens: list[str], span_tokens: list[str]) -> bool:
    """Checks whether ``span_tokens`` occurs as a contiguous slice of ``tokens``."""
    span = len(span_tokens)
    return any(tokens[i : i + span] == span_tokens for i in range(len(tokens) - span + 1))


def _token_f1(pred: str, gold: str) -> float:
    """Token F1 between two already normalized strings, ``0.0`` if either is empty."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    n_same = 0
    for token in pred_counts:
        n_same += min(pred_counts[token], gold_counts[token])
    if n_same == 0:
        return 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def context_recall(retrieved: list[Document], gold: list[SupportingFact]) -> float:
    """Fraction of gold paragraphs returned by the retrieval.

    Compares on paragraph level, so any chunk of a gold paragraph counts as a
    hit. Returns ``1.0`` for empty ``gold``.
    """
    if not gold:
        return 1.0

    gold_para_ids = {fact.paragraph_id for fact in gold}
    hit_para_ids = {chunk.metadata["paragraph_id"] for chunk in retrieved}

    n_found = 0
    for para_id in gold_para_ids:
        if para_id in hit_para_ids:
            n_found += 1
    return n_found / len(gold_para_ids)


def context_precision(retrieved: list[Document], gold: list[SupportingFact]) -> float:
    """Fraction of retrieved chunks that come from a gold paragraph.

    Returns ``0.0`` if ``retrieved`` or ``gold`` is empty.
    """
    if not retrieved or not gold:
        return 0.0

    gold_para_ids = {fact.paragraph_id for fact in gold}

    hits = 0
    for chunk in retrieved:
        if chunk.metadata["paragraph_id"] in gold_para_ids:
            hits += 1
    return hits / len(retrieved)


def answer_f1(retrieved: list[Document], gold_answer: str) -> float:
    """Best token F1 between the gold answer and a span of equal length.

    Simulates a reader that extracts the best answer span from a retrieved
    chunk, maximized over all chunks and all spans per chunk.
    """
    if not retrieved or not gold_answer:
        return 0.0

    gold_norm = _normalize_answer(gold_answer)
    gold_tokens = gold_norm.split()
    if not gold_tokens:
        return 0.0

    span_len = len(gold_tokens)
    best = 0.0
    for chunk in retrieved:
        chunk_tokens = _normalize_answer(chunk.page_content).split()
        best = max(best, _best_span_f1(chunk_tokens, gold_norm, span_len))
    return best


def exact_match(retrieved: list[Document], gold_answer: str) -> float:
    """Checks whether the gold answer appears as a token sequence in a chunk.

    Token-based instead of a plain substring check, so gold ``art`` does not
    match chunk ``apart``. Returns ``1.0`` on a hit, else ``0.0``.
    """
    gold_tokens = _normalize_answer(gold_answer).split()
    if not gold_tokens or not retrieved:
        return 0.0

    for chunk in retrieved:
        chunk_tokens = _normalize_answer(chunk.page_content).split()
        if _contains_span(chunk_tokens, gold_tokens):
            return 1.0
    return 0.0
