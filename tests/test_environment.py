"""Tests for `src.environment.environment`."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.environment import Environment


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    config = {"experiment": {"name": "test_experiment", "seed": 7}}
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "experiment_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path


def test_paths_derive_from_project_root(project_root: Path) -> None:
    env = Environment(project_root=project_root)

    assert env.raw_dir == project_root / "data" / "raw"
    assert env.processed_dir == project_root / "data" / "processed"
    assert env.data_root == project_root / "data"
    assert env.corpus_dir == project_root / "data" / "corpus"
    assert env.qa_split_path == project_root / "data" / "processed" / "qa_pairs_split.json"
    assert env.corpus_meta_path == project_root / "data" / "processed" / "corpus_meta.json"
    assert env.chroma_dir == project_root / "data" / "processed" / "chroma_db"
    assert env.results_dir == project_root / "results"
    assert env.logs_dir == env.results_dir / "logs"
    assert env.figures_dir == env.results_dir / "figures"
    assert env.tables_dir == env.results_dir / "tables"


def test_make_dirs_creates_only_given_directories(project_root: Path) -> None:
    env = Environment(project_root=project_root)

    env.make_dirs(env.raw_dir, env.processed_dir)

    assert env.raw_dir.exists()
    assert env.processed_dir.exists()
    assert not env.corpus_dir.exists()
    assert not env.results_dir.exists()


def test_set_seed_is_reproducible(project_root: Path) -> None:
    env = Environment(project_root=project_root)

    env.set_seed()
    random_a = random.random()
    numpy_a = np.random.rand()

    env.set_seed()
    random_b = random.random()
    numpy_b = np.random.rand()

    assert random_a == random_b
    assert numpy_a == numpy_b


def test_seed_and_experiment_name_come_from_config(project_root: Path) -> None:
    env = Environment(project_root=project_root)

    assert env.seed == 7
    assert env.experiment_name == "test_experiment"


def test_real_project_config_has_expected_keys() -> None:
    env = Environment()

    assert env.experiment_name
    assert isinstance(env.seed, int)
    assert env.raw_dir == env.project_root / "data" / "raw"
    assert env.processed_dir == env.project_root / "data" / "processed"
    assert env.corpus_dir == env.project_root / "data" / "corpus"
    assert env.chroma_dir == env.project_root / "data" / "processed" / "chroma_db"
