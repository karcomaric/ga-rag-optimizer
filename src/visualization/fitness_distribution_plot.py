"""Histogram of the composite fitness of all evaluations."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization._helpers import apply_style, load_history, rel_to_project, save_figure

logger = logging.getLogger(__name__)


def render_fitness_distribution(run_dir: Path) -> Path:
    """Plots the composite values as histogram into ``figures/fitness_distribution.png``."""
    df = load_history(run_dir)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    sns.histplot(df["composite"], ax=ax, color="#1f77b4")
    ax.set_xlabel("Composite-Fitness")
    ax.set_ylabel("Anzahl Evaluationen")
    ax.set_title("Verteilung der Composite-Fitness")

    out_path = save_figure(fig, run_dir, "fitness_distribution.png")
    logger.info("Abbildung geschrieben: %s", rel_to_project(out_path))
    return out_path
