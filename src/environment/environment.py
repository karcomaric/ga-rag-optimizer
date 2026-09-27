"""Environment class bundling project paths and experiment configuration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class Environment:
    """Project paths and experiment configuration for notebooks and scripts.

    The directory layout is a fixed convention of this repository, so the
    paths are built in code and only experiment parameters come from the
    YAML. Creates no directories on its own, callers use ``make_dirs``.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Loads the configuration and builds all project paths.

        ``project_root`` defaults to the directory three levels above this
        file, which keeps the paths independent of the working directory.
        """
        self.project_root: Path = project_root or Path(__file__).resolve().parents[2]
        self.config_path: Path = (
            config_path or self.project_root / "configs" / "experiment_config.yaml"
        )
        self.config: dict[str, Any] = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

        experiment = self.config["experiment"]
        self.seed: int = int(experiment["seed"])
        self.experiment_name: str = str(experiment["name"])

        self.data_root: Path = self.project_root / "data"
        self.raw_dir: Path = self.data_root / "raw"
        self.processed_dir: Path = self.data_root / "processed"
        self.corpus_dir: Path = self.data_root / "corpus"
        self.qa_split_path: Path = self.processed_dir / "qa_pairs_split.json"
        self.corpus_meta_path: Path = self.processed_dir / "corpus_meta.json"
        self.chroma_dir: Path = self.processed_dir / "chroma_db"

        self.results_dir: Path = self.project_root / "results"
        self.logs_dir: Path = self.results_dir / "logs"
        self.figures_dir: Path = self.results_dir / "figures"
        self.tables_dir: Path = self.results_dir / "tables"

    def set_seed(self) -> None:
        """Seeds Python's `random` and `numpy.random` with `self.seed`."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def make_dirs(self, *dirs: Path) -> None:
        """Creates the given directories including missing parents, existing ones stay."""
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Returns a multi-line overview of name, seed and the paths, relative to the root."""
        paths = {
            "raw_dir": self.raw_dir,
            "processed_dir": self.processed_dir,
            "corpus_dir": self.corpus_dir,
            "qa_split_path": self.qa_split_path,
            "corpus_meta_path": self.corpus_meta_path,
            "chroma_dir": self.chroma_dir,
            "results_dir": self.results_dir,
            "logs_dir": self.logs_dir,
            "figures_dir": self.figures_dir,
            "tables_dir": self.tables_dir,
        }
        lines = [f"experiment_name: {self.experiment_name}", f"seed: {self.seed}"]
        for name, path in paths.items():
            lines.append(f"{name}: {path.relative_to(self.project_root)}")
        return "\n".join(lines)
