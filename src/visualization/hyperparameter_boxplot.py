"""Boxplots of the composite fitness grouped by hyperparameter value."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from src.visualization._helpers import (
    FIGURE_DPI,
    apply_style,
    expand_individual_columns,
    load_history,
    rel_to_project,
)

logger = logging.getLogger(__name__)

BOX_COLORS = {
    "random": ("#ffbb78", "#ff7f0e"),
    "grid": ("#98df8a", "#2ca02c"),
}

GENES = ["chunk_size", "chunk_overlap", "top_k", "embedding_model", "retrieval_strategy"]

SHORT_MODEL_NAMES = {
    "all-MiniLM-L6-v2": "MiniLM",
    "BAAI/bge-small-en-v1.5": "bge-small",
    "BAAI/bge-base-en-v1.5": "bge-base",
}


def render_hyperparameter_boxplots(run_dir: Path) -> Path:
    """Plots one boxplot of the composite fitness per gene in a single row.

    The five panels share the y axis so the genes are comparable on the same
    scale. Box color follows the method (grid or random) that produced the run.
    """
    df = load_history(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    fill_color, line_color = BOX_COLORS[result["method"]]

    expand_individual_columns(df)
    df["embedding_model"] = df["embedding_model"].apply(_shorten_model_name)

    apply_style()
    fig, axes = plt.subplots(1, 5, figsize=(6.05, 2.3), sharey=True)
    for ax, gene_name in zip(axes, GENES, strict=True):
        _plot_gene(ax, df, gene_name, fill_color, line_color)

    axes[0].set_ylabel("Composite-Fitness", fontsize=7)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.1))

    model_axis = axes[GENES.index("embedding_model")]
    for label in model_axis.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")

    fig.tight_layout(w_pad=0.6)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "hyperparameter_boxplots.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    logger.info("Abbildung geschrieben: %s", rel_to_project(out_path))
    return out_path


def _shorten_model_name(name: str) -> str:
    """Maps a known embedding model name to its short label, keeps unknown names as is."""
    return SHORT_MODEL_NAMES.get(name, name)


def _plot_gene(
    ax: plt.Axes,
    df: pd.DataFrame,
    gene_name: str,
    fill_color: str,
    line_color: str,
) -> None:
    """Draws one boxplot of the composite fitness per value of a gene.

    Args:
        ax: Target axes.
        df: History with one column per gene.
        gene_name: Gene whose values form the boxes.
        fill_color: Fill of the boxes.
        line_color: Color of box edges, whiskers and medians.
    """
    sns.boxplot(
        data=df,
        x=gene_name,
        y="composite",
        ax=ax,
        color=fill_color,
        linecolor=line_color,
        linewidth=0.7,
        flierprops={"markersize": 2.5, "markeredgewidth": 0.8},
    )
    ax.set_title(gene_name, fontsize=7.5)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
