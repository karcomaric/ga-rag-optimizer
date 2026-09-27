"""GA engine: search space, operators, loop and result structure."""

from __future__ import annotations

from src.ga_engine.ga_loop import GeneticAlgorithm
from src.ga_engine.search_space import SearchSpace
from src.schemas.fitness_result import FitnessFn, FitnessResult
from src.schemas.ga_result import GAResult

__all__ = ["FitnessFn", "FitnessResult", "GAResult", "GeneticAlgorithm", "SearchSpace"]
