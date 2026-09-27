"""Derivation of the embedding-cache combinations for the prebuild step."""

from __future__ import annotations

from typing import Any

from src.ga_engine.search_space import SearchSpace
from src.rag_pipeline.pipeline import PipelineConfig, collection_name


def cache_combinations(
    search_space: SearchSpace,
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derives the deduplicated embedding-cache combinations for the prebuild.

    The default baseline combination is added even when it lies outside the
    search space grid, because it needs a cached collection just as well.
    ``top_k`` and ``retrieval_strategy`` do not enter the collection name and
    are set to arbitrary valid values.
    """
    default_combo = (
        defaults["embedding_model"],
        defaults["chunk_size"],
        defaults["chunk_overlap"],
    )
    candidates: set[tuple[str, int, int]] = set()
    for combination in search_space.full_grid():
        if not search_space.is_valid(combination):
            continue
        candidates.add(
            (
                combination["embedding_model"],
                combination["chunk_size"],
                combination["chunk_overlap"],
            )
        )
    candidates.add(default_combo)

    combinations: list[dict[str, Any]] = []
    for model, size, overlap in sorted(candidates):
        config = PipelineConfig(
            chunk_size=size,
            chunk_overlap=overlap,
            top_k=2,
            embedding_model=model,
            retrieval_strategy="dense",
        )
        combinations.append(
            {
                "collection_name": collection_name(config),
                "embedding_model": model,
                "chunk_size": size,
                "chunk_overlap": overlap,
                "is_default": (model, size, overlap) == default_combo,
            }
        )
    return combinations
