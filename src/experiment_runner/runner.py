"""ExperimentRunner class for GA-RAG optimizer runs.

Holds the run configuration as object state and orchestrates the phases of an
experiment. ``run.py`` only parses the command line and instantiates it.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from langchain_core.documents import Document

from src.baselines import (
    DefaultConfigBaseline,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)
from src.environment import Environment
from src.evaluation.aggregate import evaluate_questions, mean_metrics
from src.evaluation.fitness import composite_score
from src.evaluation.loader import load_qa_split
from src.ga_engine import GeneticAlgorithm, SearchSpace
from src.rag_pipeline.pipeline import PipelineConfig, build_retriever, load_corpus
from src.schemas import FitnessFn, FitnessResult, GAResult, OptimizerResult, QASample

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Executes a complete GA-RAG experiment run.

    The constructor loads the YAML config and applies the CLI overrides,
    ``run()`` performs the run itself.
    """

    def __init__(
        self,
        config_path: Path,
        pop_override: int | None = None,
        gen_override: int | None = None,
        qa_override: int | None = None,
        method: str = "ga",
        seed_override: int | None = None,
        trials_override: int | None = None,
    ) -> None:
        """Initializes the runner from config path and CLI overrides.

        Every ``*_override`` wins over the YAML value. ``qa_override`` stays
        ``None`` when all QA pairs of the split should be used.
        ``trials_override`` applies to random search and grid search.

        Raises:
            ValueError: If ``method`` is not one of the supported methods.
        """
        methods = ("ga", "random", "grid", "default")
        if method not in methods:
            raise ValueError(f"Unbekannte method '{method}', erwartet wird eine von {methods}.")

        self.env = Environment(config_path=config_path)
        self.config = self.env.config
        self.method = method
        self.seed = seed_override if seed_override is not None else self.env.seed
        self.population_size = (
            pop_override if pop_override is not None else int(self.config["ga"]["population_size"])
        )
        self.n_generations = (
            gen_override if gen_override is not None else int(self.config["ga"]["n_generations"])
        )
        self.qa_count = qa_override
        self.trials_override = trials_override
        self.search_space = SearchSpace(self.config["search_space"])

        self.run_dir = Path()
        self.run_id = ""

    def run(self) -> Path:
        """Executes a complete experiment run and returns the path to ``result.json``."""
        self._set_seed()
        self._create_run_dir()
        _setup_logging(self.run_dir)
        self._write_config()
        self._write_environment()
        self._log_run_info()

        documents, qa_pairs = self._load_data()
        fitness_fn = self.build_fitness_fn(documents, qa_pairs)

        logger.info("Starte %s-Lauf", self.method)
        result: GAResult | OptimizerResult
        if self.method == "ga":
            result = self._build_ga(fitness_fn).run()
        elif self.method == "random":
            result = self._build_random(fitness_fn).run()
        elif self.method == "grid":
            result = self._build_grid(fitness_fn).run()
        else:
            result = self._build_default(fitness_fn).run()

        logger.info(
            "%s abgeschlossen: best_fitness=%.4f, n_evaluations=%d, Laufzeit=%.1fs",
            self.method,
            result.best_fitness,
            result.n_evaluations,
            result.runtime_s,
        )
        logger.info("Bestes Individuum: %s", result.best_individual)

        result_path = self._save_result(result)
        logger.info("Ergebnis gespeichert unter %s", self._rel(result_path))
        self._render_figures()
        return result_path

    def _rel(self, path: Path) -> str:
        """Formats a path relative to the project root for log output."""
        try:
            return str(path.relative_to(self.env.project_root))
        except ValueError:
            return str(path)

    def _set_seed(self) -> None:
        """Sets fixed seeds for reproducibility on Python and numpy level."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _create_run_dir(self) -> None:
        """Creates the run directory and stores it in ``run_dir`` and ``run_id``.

        The run ID combines ISO timestamp and seed, so consecutive runs never
        overwrite each other.
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.run_id = f"{timestamp}_seed{self.seed}"
        self.run_dir = (
            self.env.logs_dir / self.config["experiment"]["name"] / self.method / self.run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _write_config(self) -> None:
        """Writes the configuration with the seed, population, generation and QA
        overrides applied into the run directory.

        Seed, population, generations and QA count come from the runner state,
        so the copy shows the overrides instead of the YAML defaults.
        """
        config = copy.deepcopy(self.config)
        config["experiment"]["seed"] = self.seed
        config["ga"]["population_size"] = self.population_size
        config["ga"]["n_generations"] = self.n_generations
        if self.qa_count is not None:
            config["data"]["qa_count"] = self.qa_count
        config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        (self.run_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    def _write_environment(self) -> None:
        """Writes Python and package versions into ``environment.json``.

        Without this file the software state behind the reported numbers is
        not reconstructable after the run. Packages that are not installed
        are written as ``null``.
        """
        tracked_packages = (
            "deap",
            "langchain",
            "langchain-classic",
            "langchain-community",
            "langchain-text-splitters",
            "langchain-huggingface",
            "langchain-chroma",
            "langchain-core",
            "chromadb",
            "sentence-transformers",
            "torch",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "rank-bm25",
            "numpy",
            "pandas",
        )
        packages: dict[str, str | None] = {}
        for name in tracked_packages:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None

        payload = {"python": sys.version.split()[0], "packages": packages}
        environment_text = json.dumps(payload, indent=2)
        (self.run_dir / "environment.json").write_text(environment_text, encoding="utf-8")

    def _log_run_info(self) -> None:
        """Writes the initial log lines with run metadata."""
        logger.info(
            "Starte Experiment %s, Methode %s, Run-ID %s",
            self.config["experiment"]["name"],
            self.method,
            self.run_id,
        )
        logger.info(
            "Seed=%d, qa_count=%s",
            self.seed,
            self.qa_count if self.qa_count is not None else "all",
        )

    def _load_data(self) -> tuple[list[Document], list[QASample]]:
        """Loads the corpus and the selected QA split from the paths of the ``Environment``."""
        corpus_path = self.env.corpus_dir
        qa_path = self.env.qa_split_path
        question_split = self.config["data"]["question_split"]

        documents = load_corpus(corpus_path)
        logger.info("Korpus geladen: %d Dokumente aus %s", len(documents), self._rel(corpus_path))

        qa_splits = load_qa_split(qa_path)
        qa_pairs = qa_splits[question_split]
        if self.qa_count is not None:
            qa_pairs = qa_pairs[: self.qa_count]
        logger.info("QA-Paare geladen: %d aus Split '%s'", len(qa_pairs), question_split)
        return documents, qa_pairs

    def build_fitness_fn(
        self,
        documents: list[Document],
        qa_pairs: list[QASample],
    ) -> Callable[[dict[str, Any]], FitnessResult]:
        """Builds the fitness function for the optimizer.

        The returned closure builds a retriever per chromosome, evaluates it on
        all QA pairs and averages the metrics into the composite score. The
        per-question details go into the JSONL log. Public so notebooks can
        score a stored configuration on another question split.

        Raises:
            ValueError: If ``qa_pairs`` is empty.
        """
        if not qa_pairs:
            raise ValueError("qa_pairs darf nicht leer sein.")

        chroma_dir = self.env.chroma_dir
        weights = self.config["evaluation"]["fitness_weights"]

        def fitness(individual: dict[str, Any]) -> FitnessResult:
            config = PipelineConfig(
                chunk_size=int(individual["chunk_size"]),
                chunk_overlap=int(individual["chunk_overlap"]),
                top_k=int(individual["top_k"]),
                embedding_model=str(individual["embedding_model"]),
                retrieval_strategy=str(individual["retrieval_strategy"]),
            )
            retriever = build_retriever(documents, config, chroma_dir=chroma_dir)

            per_question = evaluate_questions(retriever, qa_pairs)
            metrics = mean_metrics(per_question)
            return FitnessResult(
                composite=composite_score(metrics, weights),
                context_recall=metrics["context_recall"],
                context_precision=metrics["context_precision"],
                answer_f1=metrics["answer_f1"],
                exact_match=metrics["exact_match"],
                per_question=per_question,
            )

        return fitness

    def _build_ga(self, fitness_fn: FitnessFn) -> GeneticAlgorithm:
        """Creates a configured ``GeneticAlgorithm`` from the ``ga`` config block."""
        max_evaluations = self.population_size * (self.n_generations + 1)
        logger.info(
            "GA-Konfiguration: pop=%d, gen=%d, höchstens %d Evaluationen",
            self.population_size,
            self.n_generations,
            max_evaluations,
        )
        return GeneticAlgorithm(
            search_space=self.search_space,
            fitness_fn=fitness_fn,
            population_size=self.population_size,
            n_generations=self.n_generations,
            crossover_prob=float(self.config["ga"]["crossover_prob"]),
            mutation_prob=float(self.config["ga"]["mutation_prob"]),
            tournament_size=int(self.config["ga"]["tournament_size"]),
            mutation_indpb=float(self.config["ga"]["mutation_indpb"]),
            crossover_indpb=float(self.config["ga"]["crossover_indpb"]),
            seed=self.seed,
            log_path=self.run_dir / "ga_history.jsonl",
        )

    def _build_random(self, fitness_fn: FitnessFn) -> RandomSearchOptimizer:
        """Creates the random search baseline with the configured trial budget."""
        if self.trials_override is not None:
            n_trials = self.trials_override
        else:
            n_trials = int(self.config["baselines"]["random_search"]["n_trials"])
        logger.info("Random-Search-Konfiguration: %d Ziehungen", n_trials)
        return RandomSearchOptimizer(
            search_space=self.search_space,
            fitness_fn=fitness_fn,
            n_trials=n_trials,
            seed=self.seed,
            log_path=self.run_dir / "random_history.jsonl",
        )

    def _build_grid(self, fitness_fn: FitnessFn) -> GridSearchOptimizer:
        """Creates the grid search baseline, uncapped outside of smoke runs."""
        max_trials = self.trials_override
        if max_trials is None:
            logger.info("Grid-Search-Konfiguration: vollständiges gültiges Raster")
        else:
            logger.info("Grid-Search-Konfiguration: Raster gekappt auf %d Kandidaten", max_trials)
        return GridSearchOptimizer(
            search_space=self.search_space,
            fitness_fn=fitness_fn,
            log_path=self.run_dir / "grid_history.jsonl",
            max_trials=max_trials,
        )

    def _build_default(self, fitness_fn: FitnessFn) -> DefaultConfigBaseline:
        """Creates the default configuration baseline from the YAML values."""
        default_individual = dict(self.config["baselines"]["defaults"])
        logger.info("Default-Konfiguration: %s", default_individual)
        return DefaultConfigBaseline(
            default_individual=default_individual,
            fitness_fn=fitness_fn,
            log_path=self.run_dir / "default_history.jsonl",
        )

    def _render_figures(self) -> None:
        """Renders the figures configured for ``self.method``.

        A failing figure does not abort the run, because the result is already
        saved when this runs. The import is local because ``src.visualization``
        switches the matplotlib backend to ``Agg`` on import, which would break
        inline plotting in the notebooks that import this module.
        """
        from src.visualization import (
            render_convergence,
            render_fitness_distribution,
            render_hyperparameter_boxplots,
            render_metric_components,
            render_search_progress,
        )

        figures: tuple[Callable[[Path], Path], ...]
        if self.method == "ga":
            figures = (
                render_convergence,
                render_search_progress,
                render_fitness_distribution,
                render_metric_components,
            )
        elif self.method == "random":
            figures = (
                render_fitness_distribution,
                render_search_progress,
                render_hyperparameter_boxplots,
                render_metric_components,
            )
        elif self.method == "grid":
            figures = (
                render_hyperparameter_boxplots,
                render_fitness_distribution,
                render_search_progress,
                render_metric_components,
            )
        else:
            figures = (render_metric_components,)

        for render in figures:
            try:
                render(self.run_dir)
            except Exception:
                logger.exception(
                    "Rendering von %s für %s fehlgeschlagen",
                    render.__name__,
                    self._rel(self.run_dir),
                )

    def _save_result(self, result: GAResult | OptimizerResult) -> Path:
        """Writes the result as ``result.json`` into the run directory.

        Method, seed and run ID are added so the file stays readable without
        parsing the directory name.
        """
        data: dict[str, Any] = {
            "method": self.method,
            "seed": self.seed,
            "run_id": self.run_id,
        }
        data.update(asdict(result))
        result_path = self.run_dir / "result.json"
        result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return result_path


def _setup_logging(run_dir: Path) -> None:
    """Configures console and file logging into the given directory.

    Handlers of a previous run are closed before they are dropped, otherwise
    repeated runs in one process leak open log files. Talkative third party
    loggers are set to ``WARNING`` so HTTP requests from HuggingFace and
    ChromaDB do not flood ``experiment.log``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    noisy_loggers = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "urllib3",
        "chromadb",
        "matplotlib",
    )
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
