"""Convergence plot of a GA run."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization._helpers import apply_style, rel_to_project, save_figure

logger = logging.getLogger(__name__)


def render_convergence(run_dir: Path) -> Path:
    """Plots the mean population fitness per generation with a min-max band.

    Reads the logbook from ``result.json``. The band spans the measured minimum
    and maximum, so it cannot reach beyond values the run actually produced.

    Raises:
        FileNotFoundError: If ``result.json`` is missing in the run directory.
    """
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json fehlt im Run-Verzeichnis: {result_path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(result["logbook"])

    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    gen = df["gen"]
    ax.plot(gen, df["avg"], label="Mittelwert", color="#1f77b4", linewidth=2)
    ax.fill_between(
        gen,
        df["min"],
        df["max"],
        alpha=0.2,
        color="#1f77b4",
        label="Minimum bis Maximum",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Composite-Fitness")
    ax.set_title("Konvergenz des GA-Laufs")
    ax.legend(loc="upper left", framealpha=0.9)

    out_path = save_figure(fig, run_dir, "convergence.png")
    logger.info("Abbildung geschrieben: %s", rel_to_project(out_path))
    return out_path
