"""Experiment runner entry point for GA-RAG optimizer experiments.

Usage:
    python src/experiment_runner/run.py --config configs/experiment_config.yaml
    python src/experiment_runner/run.py --config configs/experiment_config.yaml --smoke
    python src/experiment_runner/run.py --config configs/experiment_config.yaml --pop 5 --gen 5

The run logic itself lives in ``src.experiment_runner.runner``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment_runner.runner import ExperimentRunner


def parse_cli_args() -> argparse.Namespace:
    """Reads the command line arguments and returns them as a namespace."""
    parser = argparse.ArgumentParser(description="GA-RAG-Optimizer Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast debug mode with a small population, few generations, few QA pairs.",
    )
    parser.add_argument("--pop", type=int, default=None, help="Override population_size.")
    parser.add_argument("--gen", type=int, default=None, help="Override n_generations.")
    parser.add_argument("--qa", type=int, default=None, help="Override the QA pair count.")
    parser.add_argument(
        "--method",
        type=str,
        choices=("ga", "random", "grid", "default"),
        default="ga",
        help="Optimization method to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override the trial count for random search and grid search.",
    )
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    """Fills the overrides that were not set on the command line with small debug values."""
    if args.pop is None:
        args.pop = 3
    if args.gen is None:
        args.gen = 2
    if args.qa is None:
        args.qa = 5
    if args.trials is None:
        args.trials = 6


def main() -> None:
    """Main entry point, parses the CLI and starts the runner."""
    args = parse_cli_args()
    if args.smoke:
        apply_smoke_defaults(args)
    runner = ExperimentRunner(
        config_path=Path(args.config).resolve(),
        pop_override=args.pop,
        gen_override=args.gen,
        qa_override=args.qa,
        method=args.method,
        seed_override=args.seed,
        trials_override=args.trials,
    )
    runner.run()


if __name__ == "__main__":
    main()
