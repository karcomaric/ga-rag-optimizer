"""Shared helpers for the plot functions.

Sets the matplotlib backend to ``Agg``, which renders into files without a
GUI toolkit. The default here would be ``TkAgg``, whose tkinter setup fails
when the full test suite runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"


def apply_style() -> None:
    """Activates the project figure style, falling back to ``default`` on error."""
    try:
        plt.style.use(FIGURE_STYLE)
    except (OSError, ValueError):
        logger.warning("Stil %s nicht verfügbar, Abbildung nutzt den Standardstil.", FIGURE_STYLE)
        plt.style.use("default")


def save_figure(fig: plt.Figure, run_dir: Path, filename: str) -> Path:
    """Saves a figure as PNG under ``run_dir/figures/`` and closes it."""
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def rel_to_project(path: Path) -> str:
    """Formats a path relative to the project root for log output."""
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_history(run_dir: Path) -> pd.DataFrame:
    """Loads the ``*_history.jsonl`` of a run as one row per evaluation.

    Raises:
        FileNotFoundError: If no ``*_history.jsonl`` file exists in ``run_dir``.
    """
    matches = sorted(run_dir.glob("*_history.jsonl"))
    if not matches:
        raise FileNotFoundError(f"Keine *_history.jsonl-Datei gefunden in {run_dir}")
    return pd.read_json(matches[0], lines=True)


def expand_individual_columns(df: pd.DataFrame) -> list[str]:
    """Extends ``df`` in place with one column per gene and returns their names."""
    expanded = pd.json_normalize(df["individual"])
    for column in expanded.columns:
        df[column] = expanded[column].values
    return list(expanded.columns)
