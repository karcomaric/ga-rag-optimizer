"""Tests for ``src.visualization``.

Checks existence and minimum size of the generated PNGs, no pixel diffs.
The fixture creates a mini run directory with a synthetic logbook and
JSONL history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.visualization import (
    render_convergence,
    render_fitness_distribution,
    render_hyperparameter_boxplots,
    render_metric_components,
    render_search_progress,
)
from src.visualization._helpers import load_history, rel_to_project

MIN_PNG_BYTES = 1024


def test_rel_to_project_outside_project(tmp_path: Path) -> None:
    outside_path = tmp_path / "convergence.png"
    assert rel_to_project(outside_path) == str(outside_path)


def _individual(chunk_size: int, top_k: int, strategy: str) -> dict[str, object]:
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": 32,
        "top_k": top_k,
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_strategy": strategy,
    }


def _history_record(eval_index: int, composite: float, individual: dict[str, object]) -> dict:
    return {
        "eval_index": eval_index,
        "individual": individual,
        "runtime_s": 0.05,
        "timestamp": "2026-05-27T10:00:00",
        "composite": composite,
        "context_recall": composite,
        "context_precision": composite,
        "answer_f1": composite,
        "exact_match": composite,
    }


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Creates a run directory with a minimal logbook and JSONL history."""
    run_dir = tmp_path / "run_2026-05-27"
    run_dir.mkdir()

    logbook = [
        {"gen": 0, "nevals": 4, "avg": 0.30, "std": 0.05, "min": 0.20, "max": 0.40},
        {"gen": 1, "nevals": 3, "avg": 0.45, "std": 0.06, "min": 0.30, "max": 0.55},
        {"gen": 2, "nevals": 3, "avg": 0.60, "std": 0.04, "min": 0.50, "max": 0.70},
    ]
    result = {
        "method": "grid",
        "best_individual": _individual(1024, 5, "hybrid"),
        "best_fitness": 0.70,
        "logbook": logbook,
        "n_evaluations": 8,
        "runtime_s": 12.3,
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    history = [
        _history_record(0, 0.20, _individual(128, 1, "dense")),
        _history_record(1, 0.30, _individual(128, 3, "bm25")),
        _history_record(2, 0.40, _individual(512, 3, "dense")),
        _history_record(3, 0.45, _individual(512, 5, "hybrid")),
        _history_record(4, 0.50, _individual(512, 5, "hybrid")),
        _history_record(5, 0.55, _individual(1024, 3, "hybrid")),
        _history_record(6, 0.65, _individual(1024, 5, "hybrid")),
        _history_record(7, 0.70, _individual(1024, 5, "hybrid")),
    ]
    history_path = run_dir / "ga_history.jsonl"
    history_path.write_text(
        "\n".join(json.dumps(record) for record in history) + "\n", encoding="utf-8"
    )
    return run_dir


def _assert_png(path: Path) -> None:
    assert path.exists(), f"PNG fehlt: {path}"
    assert path.stat().st_size > MIN_PNG_BYTES, f"PNG zu klein: {path.stat().st_size} Bytes"


def test_convergence_plot_runs(tmp_run_dir: Path) -> None:
    out = render_convergence(tmp_run_dir)
    assert out == tmp_run_dir / "figures" / "convergence.png"
    _assert_png(out)


def test_fitness_distribution_plot_runs(tmp_run_dir: Path) -> None:
    out = render_fitness_distribution(tmp_run_dir)
    assert out == tmp_run_dir / "figures" / "fitness_distribution.png"
    _assert_png(out)


def test_hyperparameter_boxplot_runs(tmp_run_dir: Path) -> None:
    out = render_hyperparameter_boxplots(tmp_run_dir)
    assert out == tmp_run_dir / "figures" / "hyperparameter_boxplots.png"
    _assert_png(out)


def test_search_progress_plot_runs(tmp_run_dir: Path) -> None:
    out = render_search_progress(tmp_run_dir)
    assert out == tmp_run_dir / "figures" / "search_progress.png"
    _assert_png(out)


def test_metric_components_plot_runs(tmp_run_dir: Path) -> None:
    out = render_metric_components(tmp_run_dir)
    assert out == tmp_run_dir / "figures" / "metric_components.png"
    _assert_png(out)


def test_load_history_finds_non_ga_history_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "random_run"
    run_dir.mkdir()
    history = [_history_record(0, 0.5, _individual(128, 1, "dense"))]
    (run_dir / "random_history.jsonl").write_text(
        "\n".join(json.dumps(record) for record in history) + "\n", encoding="utf-8"
    )

    df = load_history(run_dir)
    assert len(df) == 1


def test_load_history_raises_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_history(run_dir)
