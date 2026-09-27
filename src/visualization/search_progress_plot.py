"""Scatter plot of all evaluations with the cumulative best value."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization._helpers import apply_style, load_history, rel_to_project, save_figure

logger = logging.getLogger(__name__)


def render_search_progress(run_dir: Path) -> Path:
    """Plots all composite values as scatter with the running maximum as line.

    Saves as ``figures/search_progress.png``.
    """
    df = load_history(run_dir)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    evaluations = range(1, len(df) + 1)
    best_so_far = df["composite"].cummax()
    ax.scatter(evaluations, df["composite"], alpha=0.3, color="#1f77b4", label="Evaluation")
    ax.plot(
        evaluations,
        best_so_far,
        color="#d62728",
        linewidth=2,
        label="Bestes Ergebnis bisher",
    )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Composite-Fitness")
    ax.set_title("Suchfortschritt: bestes Ergebnis über die Evaluationen")
    ax.legend(loc="lower right", framealpha=0.9)

    out_path = save_figure(fig, run_dir, "search_progress.png")
    logger.info("Abbildung geschrieben: %s", rel_to_project(out_path))
    return out_path
