"""Tests for ``src.rag_pipeline.pipeline``."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.rag_pipeline.pipeline import (
    PipelineConfig,
    _get_embeddings,
    build_retriever,
    collection_name,
    load_corpus,
)


def _write_corpus(corpus_dir: Path, items: dict[str, str]) -> None:
    """Writes one .txt file per entry in the expected format."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for paragraph_id, body in items.items():
        path = corpus_dir / f"{paragraph_id}.txt"
        path.write_text(f"{paragraph_id}\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def short_corpus(tmp_path: Path) -> list[Document]:
    """Three one-sentence paragraphs with the IDs p1, p2 and p3."""
    corpus = tmp_path / "corpus"
    _write_corpus(
        corpus,
        {
            "p1": "Berlin is the capital of Germany.",
            "p2": "Paris is the capital of France.",
            "p3": "Rome is the capital of Italy.",
        },
    )
    return load_corpus(corpus)


@pytest.fixture
def city_corpus(tmp_path: Path) -> list[Document]:
    """Three city paragraphs long enough for a retrieval query to discriminate."""
    corpus = tmp_path / "corpus"
    _write_corpus(
        corpus,
        {
            "berlin": "Berlin is the capital of Germany and a large European city.",
            "paris": "Paris is the capital of France and famous for the Eiffel Tower.",
            "rome": "Rome is the capital of Italy and the seat of the Vatican.",
        },
    )
    return load_corpus(corpus)


def test_load_corpus_yields_one_document_per_file(short_corpus: list[Document]) -> None:
    docs = short_corpus

    assert len(docs) == 3
    paragraph_ids = {d.metadata["paragraph_id"] for d in docs}
    assert paragraph_ids == {"p1", "p2", "p3"}
    assert all(d.metadata["title"] in {"p1", "p2", "p3"} for d in docs)
    assert all(isinstance(d.page_content, str) and d.page_content for d in docs)


def test_load_corpus_raises_for_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist")


def test_hp_config_rejects_overlap_greater_equal_size() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            chunk_size=128,
            chunk_overlap=128,
            top_k=5,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            retrieval_strategy="dense",
        )


def test_collection_name_replaces_slashes_and_embeds_chunk_params() -> None:
    config = PipelineConfig(
        chunk_size=256,
        chunk_overlap=64,
        top_k=1,
        embedding_model="BAAI/bge-small-en-v1.5",
        retrieval_strategy="dense",
    )
    assert collection_name(config) == "BAAI_bge-small-en-v1.5_cs256_co64"


def test_paragraph_id_preserved_after_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    long_body = " ".join([f"Sentence number {i}." for i in range(40)])
    _write_corpus(corpus, {"long_paragraph": long_body})
    docs = load_corpus(corpus)

    config = PipelineConfig(
        chunk_size=120,
        chunk_overlap=20,
        top_k=3,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="bm25",
    )
    retriever = build_retriever(docs, config, chroma_dir=tmp_path / "chroma")
    results = retriever.invoke("Sentence number 5")

    assert results, "BM25 muss für eine passende Anfrage mindestens einen Chunk liefern"
    assert all(r.metadata["paragraph_id"] == "long_paragraph" for r in results)


def test_build_bm25_retriever_returns_top_k_documents(
    city_corpus: list[Document], tmp_path: Path
) -> None:
    config = PipelineConfig(
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="bm25",
    )
    retriever = build_retriever(city_corpus, config, chroma_dir=tmp_path / "chroma")
    results = retriever.invoke("Eiffel Tower")

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].metadata["paragraph_id"] == "paris"


def test_dense_collection_is_reused_on_second_build(
    short_corpus: list[Document], tmp_path: Path
) -> None:
    """Reproducibility: a second call with the same config must not duplicate the collection."""
    import chromadb

    config = PipelineConfig(
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="dense",
    )
    chroma_dir = tmp_path / "chroma"

    _ = build_retriever(short_corpus, config, chroma_dir=chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    count_after_first = client.list_collections()[0].count()

    _ = build_retriever(short_corpus, config, chroma_dir=chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    count_after_second = client.list_collections()[0].count()

    assert count_after_first == count_after_second, (
        f"Collection must not grow on the second build, "
        f"was {count_after_first}, is now {count_after_second}"
    )


def test_build_hybrid_retriever_returns_at_most_top_k_documents(
    city_corpus: list[Document], tmp_path: Path
) -> None:
    """Hybrid fuses BM25 and dense via RRF and is truncated to top_k."""
    config = PipelineConfig(
        chunk_size=200,
        chunk_overlap=0,
        top_k=2,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="hybrid",
    )
    retriever = build_retriever(city_corpus, config, chroma_dir=tmp_path / "chroma")
    results = retriever.invoke("Which city is famous for the Eiffel Tower?")

    assert 0 < len(results) <= 2, f"Hybrid darf höchstens 2 Chunks liefern, waren {len(results)}"
    assert all(isinstance(r, Document) for r in results)
    assert all("paragraph_id" in r.metadata for r in results)


def test_embeddings_are_loaded_once_per_model() -> None:
    first = _get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    second = _get_embeddings("sentence-transformers/all-MiniLM-L6-v2")

    assert first is second, "Ein zweiter Aufruf muss dasselbe Modell-Objekt liefern"


def test_build_dense_retriever_returns_top_k_documents(
    city_corpus: list[Document], tmp_path: Path
) -> None:
    config = PipelineConfig(
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="dense",
    )
    retriever = build_retriever(city_corpus, config, chroma_dir=tmp_path / "chroma")
    results = retriever.invoke("Which city is famous for the Eiffel Tower?")

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].metadata["paragraph_id"] == "paris"
