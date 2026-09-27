"""Tests for ``src.ga_engine.search_space``."""

from __future__ import annotations

import random

import pytest

from src.ga_engine.search_space import SearchSpace


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        {
            "chunk_size": [128, 512, 1024],
            "chunk_overlap": [0, 32, 64, 128],
            "top_k": [2, 3, 5, 10],
            "embedding_model": [
                "all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
            ],
            "retrieval_strategy": ["dense", "bm25", "hybrid"],
        }
    )


def test_full_grid_covers_all_values_with_last_gene_fastest() -> None:
    space = SearchSpace({"a": [1, 2], "b": ["x", "y", "z"]})

    combinations = space.full_grid()

    assert len(combinations) == 6
    assert combinations[0] == {"a": 1, "b": "x"}
    assert combinations[1] == {"a": 1, "b": "y"}
    assert combinations[-1] == {"a": 2, "b": "z"}


def test_search_space_holds_all_five_genes(space: SearchSpace) -> None:
    assert set(space.gene_names) == {
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "embedding_model",
        "retrieval_strategy",
    }


def test_sample_is_reproducible_with_seed(space: SearchSpace) -> None:
    random.seed(42)
    sample_a = space.sample()
    random.seed(42)
    sample_b = space.sample()

    assert sample_a == sample_b


def test_sample_returns_value_from_each_axis(space: SearchSpace) -> None:
    random.seed(0)
    sample = space.sample()

    for gene, value in sample.items():
        assert value in space.values[gene]


def test_mutate_gene_at_lower_bound(space: SearchSpace) -> None:
    for _ in range(20):
        assert space.mutate_gene("chunk_size", 128) == 512


def test_mutate_gene_at_upper_bound(space: SearchSpace) -> None:
    for _ in range(20):
        assert space.mutate_gene("chunk_overlap", 128) == 64


def test_mutate_categorical_always_picks_different_value(space: SearchSpace) -> None:
    for _ in range(50):
        mutated = space.mutate_gene("retrieval_strategy", "dense")
        assert mutated != "dense"
        assert mutated in {"bm25", "hybrid"}


def test_sample_never_violates_overlap_constraint(space: SearchSpace) -> None:
    random.seed(42)
    for _ in range(200):
        individual = space.sample()
        assert individual["chunk_overlap"] < individual["chunk_size"]


def test_repair_repairs_invalid_individual(space: SearchSpace) -> None:
    invalid = {"chunk_size": 128, "chunk_overlap": 128}
    random.seed(42)
    fixed = space.repair(invalid)

    assert fixed["chunk_overlap"] < fixed["chunk_size"]
    assert fixed["chunk_overlap"] in (0, 32, 64)


def test_repair_leaves_valid_individual_unchanged(space: SearchSpace) -> None:
    valid = {"chunk_size": 512, "chunk_overlap": 64, "top_k": 5}
    fixed = space.repair(valid)

    assert fixed == valid


def test_is_valid_only_accepts_overlap_below_chunk_size(space: SearchSpace) -> None:
    assert space.is_valid({"chunk_size": 512, "chunk_overlap": 64}) is True
    assert space.is_valid({"chunk_size": 128, "chunk_overlap": 128}) is False
