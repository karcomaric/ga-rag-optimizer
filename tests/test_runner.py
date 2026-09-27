"""Tests for ``src.experiment_runner.runner``."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.baselines import OptimizerResult
from src.experiment_runner.runner import ExperimentRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"


def test_rel_inside_project_root() -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH)
    result_path = runner.env.project_root / "results" / "logs" / "result.json"
    assert runner._rel(result_path) == str(Path("results") / "logs" / "result.json")


def test_rel_outside_project_root(tmp_path: Path) -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH)
    outside_path = tmp_path / "result.json"
    assert runner._rel(outside_path) == str(outside_path)


def test_write_config_documents_effective_values(tmp_path: Path) -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH, seed_override=99, qa_override=5)
    runner.run_dir = tmp_path
    runner._write_config()

    written = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert written["experiment"]["seed"] == 99
    assert written["ga"]["population_size"] == runner.population_size
    assert written["ga"]["n_generations"] == runner.n_generations
    assert written["data"]["qa_count"] == runner.qa_count


def test_write_environment_records_python_and_packages(tmp_path: Path) -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH)
    runner.run_dir = tmp_path
    runner._write_environment()

    written = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert written["python"].startswith("3.")
    assert written["packages"]["deap"] is not None
    assert written["packages"]["langchain-classic"] is not None
    assert written["packages"]["chromadb"] is not None
    assert written["packages"]["sentence-transformers"] is not None


def test_trials_override_applies_to_random_and_grid(tmp_path: Path) -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH, trials_override=24)
    runner.run_dir = tmp_path

    random_opt = runner._build_random(lambda individual: None)
    grid_opt = runner._build_grid(lambda individual: None)

    assert random_opt.n_trials == 24
    assert grid_opt.max_trials == 24


def test_save_result_adds_method_seed_and_run_id(tmp_path: Path) -> None:
    runner = ExperimentRunner(config_path=CONFIG_PATH, method="random", seed_override=7)
    runner.run_dir = tmp_path
    runner.run_id = tmp_path.name
    result = OptimizerResult(
        best_individual={"chunk_size": 128},
        best_fitness=0.5,
        n_evaluations=1,
        runtime_s=0.1,
    )

    result_path = runner._save_result(result)

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["method"] == "random"
    assert data["seed"] == 7
    assert data["run_id"] == tmp_path.name
    assert data["best_fitness"] == 0.5
