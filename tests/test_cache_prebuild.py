"""Tests for the embedding-cache combination derivation."""

from __future__ import annotations

from src.experiment_runner.cache_prebuild import cache_combinations
from src.ga_engine.search_space import SearchSpace


def test_cache_combinations_filters_dedupes_and_flags_default() -> None:
    space = SearchSpace(
        {
            "chunk_size": [128, 256],
            "chunk_overlap": [0, 128],
            "embedding_model": ["all-MiniLM-L6-v2"],
        }
    )
    defaults = {"embedding_model": "all-MiniLM-L6-v2", "chunk_size": 256, "chunk_overlap": 0}

    combos = cache_combinations(space, defaults)

    assert [c["collection_name"] for c in combos] == [
        "all-MiniLM-L6-v2_cs128_co0",
        "all-MiniLM-L6-v2_cs256_co0",
        "all-MiniLM-L6-v2_cs256_co128",
    ]
    assert [c["is_default"] for c in combos] == [False, True, False]


def test_cache_combinations_appends_out_of_grid_default() -> None:
    space = SearchSpace(
        {
            "chunk_size": [128],
            "chunk_overlap": [0],
            "embedding_model": ["all-MiniLM-L6-v2"],
        }
    )
    defaults = {"embedding_model": "all-MiniLM-L6-v2", "chunk_size": 1000, "chunk_overlap": 200}

    combos = cache_combinations(space, defaults)

    assert len(combos) == 2
    assert combos[1]["collection_name"] == "all-MiniLM-L6-v2_cs1000_co200"
    assert combos[1]["is_default"] is True
