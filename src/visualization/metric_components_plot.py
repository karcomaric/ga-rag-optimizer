"""Bar chart of the fitness components of the best individual of a run."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization._helpers import apply_style, load_history, rel_to_project, save_figure

logger = logging.getLogger(__name__)


def render_metric_components(run_dir: Path) -> Path:
    """Plots the metric components of the best individual of a run as bars.

    Three colors separate the metrics that make up the composite, the composite
    itself, and exact match, which is logged but not part of the fitness.
    """
    df = load_history(run_dir)
    best_row = df.loc[df["composite"].idxmax()]

    labels = ["Context Recall", "Context Precision", "Answer-F1", "Exact Match", "Composite"]
    values = [
        float(best_row["context_recall"]),
        float(best_row["context_precision"]),
        float(best_row["answer_f1"]),
        float(best_row["exact_match"]),
        float(best_row["composite"]),
    ]
    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#7f7f7f", "#d62728"]

    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metrikwert")
    ax.set_title("Fitness-Komponenten des besten Individuums")
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_ha("right")

    out_path = save_figure(fig, run_dir, "metric_components.png")
    logger.info("Abbildung geschrieben: %s", rel_to_project(out_path))
    return out_path
