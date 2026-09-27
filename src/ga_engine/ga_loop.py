"""Genetic algorithm based on DEAP.

Thin shell around ``deap.algorithms.eaSimple``. The class configures the
toolbox, owns crossover and mutation and writes one JSONL line per fitness
call. Selection and main loop come directly from DEAP. The per-generation
statistics are registered on the DEAP ``Statistics`` object and cover both
fitness and genotypic diversity.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from deap import algorithms, base, tools

from src.ga_engine.search_space import SearchSpace
from src.schemas.fitness_result import FitnessFn
from src.schemas.ga_result import GAResult

logger = logging.getLogger(__name__)


class FitnessMax(base.Fitness):
    """Single objective fitness that DEAP maximises."""

    weights = (1.0,)


class Individual(dict):
    """Dict chromosome carrying the fitness object that DEAP expects."""

    def __init__(self, genes: dict[str, Any]) -> None:
        super().__init__(genes)
        self.fitness = FitnessMax()


class GeneticAlgorithm:
    """GA loop with DEAP, dict chromosomes and a JSONL evaluation log.

    Owns the toolbox, the operators, the seeding and the logging of one line
    per fitness call. Crossover and mutation are static, so they can be used
    and tested without an instance. The main loop itself runs entirely in
    ``deap.algorithms.eaSimple``.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: FitnessFn,
        population_size: int,
        n_generations: int,
        crossover_prob: float,
        mutation_prob: float,
        tournament_size: int,
        mutation_indpb: float,
        crossover_indpb: float,
        seed: int,
        log_path: Path | None = None,
    ) -> None:
        """Seeds ``random`` and ``numpy`` and builds the toolbox.

        ``crossover_prob`` and ``mutation_prob`` are ``cxpb`` and ``mutpb`` for
        ``eaSimple``, ``mutation_indpb`` and ``crossover_indpb`` are the
        per-gene probabilities for mutation and swap. ``log_path`` of ``None``
        disables the JSONL log.
        """
        random.seed(seed)
        np.random.seed(seed)

        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.mutation_indpb = mutation_indpb
        self.crossover_indpb = crossover_indpb
        self.log_path = log_path
        self._eval_index = 0
        self._generation = 0
        self._best_seen = float("-inf")
        self._log_file: TextIO | None = None

        self.toolbox = self._build_toolbox()

    def _fitness_values(self, population: tuple[Any, ...]) -> list[float]:
        """Reads the composite fitness of every individual in the population."""
        values: list[float] = []
        for individual in population:
            values.append(float(individual.fitness.values[0]))
        return values

    def _mean_fitness(self, population: tuple[Any, ...]) -> float:
        """Mean composite fitness of the population."""
        return float(np.mean(self._fitness_values(population)))

    def _std_fitness(self, population: tuple[Any, ...]) -> float:
        """Standard deviation of the composite fitness in the population."""
        return float(np.std(self._fitness_values(population)))

    def _min_fitness(self, population: tuple[Any, ...]) -> float:
        """Lowest composite fitness in the population."""
        return float(np.min(self._fitness_values(population)))

    def _max_fitness(self, population: tuple[Any, ...]) -> float:
        """Highest composite fitness in the population."""
        return float(np.max(self._fitness_values(population)))

    def _unique_genomes(self, population: tuple[Any, ...]) -> int:
        """Counts the distinct chromosomes in the population."""
        genomes: set[str] = set()
        for individual in population:
            genomes.add(json.dumps(dict(individual), sort_keys=True))
        return len(genomes)

    def _gene_diversity(self, population: tuple[Any, ...]) -> float:
        """Mean number of distinct values per gene across the population."""
        counts: list[int] = []
        for gene in population[0].keys():
            values: set[Any] = set()
            for individual in population:
                values.add(individual[gene])
            counts.append(len(values))
        return float(np.mean(counts))

    def _init_individual(self) -> Individual:
        """Draws one valid individual from the search space."""
        return Individual(self.search_space.sample())

    def _select(self, population: list[Any], k: int) -> list[Any]:
        """Picks ``k`` parents by tournament and counts the generation.

        ``eaSimple`` calls the selection once at the start of every
        generation, which makes this the only place where the generation
        number is known before its evaluations happen.
        """
        self._generation += 1
        return tools.selTournament(population, k, tournsize=self.tournament_size)

    @staticmethod
    def uniform_crossover(
        parent_a: dict[str, Any],
        parent_b: dict[str, Any],
        search_space: SearchSpace,
        indpb: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Uniform crossover for dict chromosomes, swapping each gene with ``indpb``.

        The parents stay unchanged. Swapping can combine values that violate
        ``chunk_overlap < chunk_size``, so both children are repaired.
        """
        child_a: dict[str, Any] = dict(parent_a)
        child_b: dict[str, Any] = dict(parent_b)
        for gene in parent_a:
            if random.random() < indpb:
                child_a[gene], child_b[gene] = child_b[gene], child_a[gene]
        return search_space.repair(child_a), search_space.repair(child_b)

    @staticmethod
    def mutate_individual(
        individual: dict[str, Any],
        search_space: SearchSpace,
        indpb: float,
    ) -> dict[str, Any]:
        """Redraws each gene with probability ``indpb`` via ``search_space.mutate_gene``.

        The input stays unchanged. Mutating ``chunk_size`` and
        ``chunk_overlap`` independently can violate the chunk constraint, so
        the result is repaired.
        """
        mutant: dict[str, Any] = dict(individual)
        for gene, value in individual.items():
            if random.random() < indpb:
                mutant[gene] = search_space.mutate_gene(gene, value)
        return search_space.repair(mutant)

    def _mate(self, ind_a: Any, ind_b: Any) -> tuple[Individual, Individual]:
        """Crosses two individuals and returns the two children DEAP expects."""
        child_a, child_b = self.uniform_crossover(
            dict(ind_a), dict(ind_b), self.search_space, self.crossover_indpb
        )
        return Individual(child_a), Individual(child_b)

    def _mutate(self, ind: Any) -> tuple[Individual]:
        """Mutates one individual and wraps it in the single element tuple DEAP expects."""
        mutant = self.mutate_individual(dict(ind), self.search_space, self.mutation_indpb)
        return (Individual(mutant),)

    def _build_toolbox(self) -> base.Toolbox:
        """Builds the DEAP toolbox with all GA operators."""
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", self._select)
        toolbox.register("mate", self._mate)
        toolbox.register("mutate", self._mutate)
        toolbox.register("evaluate", self._evaluate)
        return toolbox

    def _evaluate(self, individual: Any) -> tuple[float]:
        """Evaluates one individual, logs a JSONL line and returns the DEAP tuple."""
        start = time.perf_counter()
        metrics = self.fitness_fn(dict(individual))
        wall = time.perf_counter() - start

        if self._log_file is not None:
            record: dict[str, Any] = {
                "eval_index": self._eval_index,
                "generation": self._generation,
                "individual": dict(individual),
                "runtime_s": wall,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            record.update(metrics)
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

        composite = float(metrics["composite"])
        if composite > self._best_seen:
            self._best_seen = composite
        self._eval_index += 1
        logger.info(
            "Evaluation %d: composite=%.4f, bestes bisher=%.4f",
            self._eval_index,
            composite,
            self._best_seen,
        )
        return (composite,)

    def run(self) -> GAResult:
        """Runs the GA and collects the result of all generations."""
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("w", encoding="utf-8")

        stats = tools.Statistics()
        stats.register("avg", self._mean_fitness)
        stats.register("std", self._std_fitness)
        stats.register("min", self._min_fitness)
        stats.register("max", self._max_fitness)
        stats.register("unique_genomes", self._unique_genomes)
        stats.register("gene_diversity", self._gene_diversity)

        hof = tools.HallOfFame(maxsize=1)

        start = time.perf_counter()
        try:
            pop = self.toolbox.population(n=self.population_size)
            _, logbook = algorithms.eaSimple(
                pop,
                self.toolbox,
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob,
                ngen=self.n_generations,
                stats=stats,
                halloffame=hof,
                verbose=False,
            )
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
        wall = time.perf_counter() - start

        best = hof[0]
        return GAResult(
            best_individual=dict(best),
            best_fitness=float(best.fitness.values[0]),
            logbook=[dict(record) for record in logbook],
            n_evaluations=self._eval_index,
            runtime_s=wall,
        )
