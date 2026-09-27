"""Visualization for GA-RAG runs.

Reads only the result files of a finished run (``result.json``,
``*_history.jsonl``) and writes the plots into that run directory.
"""

from __future__ import annotations

from src.visualization.convergence_plot import render_convergence
from src.visualization.fitness_distribution_plot import render_fitness_distribution
from src.visualization.hyperparameter_boxplot import render_hyperparameter_boxplots
from src.visualization.metric_components_plot import render_metric_components
from src.visualization.search_progress_plot import render_search_progress

__all__ = [
    "render_convergence",
    "render_fitness_distribution",
    "render_hyperparameter_boxplots",
    "render_metric_components",
    "render_search_progress",
]
