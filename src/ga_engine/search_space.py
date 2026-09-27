"""Definition and sampling of the hyperparameter search space."""

from __future__ import annotations

import random
from typing import Any


class SearchSpace:
    """Hyperparameter search space from the YAML configuration.

    Holds the allowed values per gene, int-valued genes sorted, str-valued
    genes in the order they appear in the YAML.
    """

    def __init__(self, values: dict[str, list[Any]]) -> None:
        """Creates a search space from a mapping of gene name to value list.

        Raises:
            ValueError: If a gene value list is empty.
        """
        prepared: dict[str, list[Any]] = {}
        for gene, vals in values.items():
            if not vals:
                raise ValueError(f"Gene '{gene}' hat keine erlaubten Werte.")

            if isinstance(vals[0], int):
                prepared[gene] = sorted(vals)
            else:
                prepared[gene] = list(vals)
        self.values: dict[str, list[Any]] = prepared

    @property
    def gene_names(self) -> tuple[str, ...]:
        """Returns all gene names in YAML order."""
        return tuple(self.values.keys())

    def full_grid(self) -> list[dict[str, Any]]:
        """Builds every combination of gene values, the last gene varying fastest.

        Includes combinations that violate the chunk constraint, filtering is
        left to the caller via ``is_valid``.
        """
        combinations: list[dict[str, Any]] = [{}]
        for gene in self.gene_names:
            new_combinations: list[dict[str, Any]] = []
            for combination in combinations:
                for value in self.values[gene]:
                    new_combination = dict(combination)
                    new_combination[gene] = value
                    new_combinations.append(new_combination)
            combinations = new_combinations
        return combinations

    def sample(self) -> dict[str, Any]:
        """Draws one random value per gene, repaired so the chunk constraint holds."""
        individual: dict[str, Any] = {}
        for gene, vals in self.values.items():
            individual[gene] = random.choice(vals)
        return self.repair(individual)

    def repair(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Repairs an individual so that ``chunk_overlap`` stays below ``chunk_size``.

        Returns a new dict, with ``chunk_overlap`` redrawn from the allowed
        values below ``chunk_size`` if the constraint was violated.
        """
        fixed = dict(individual)
        if self.is_valid(fixed):
            return fixed

        valid_overlaps: list[int] = []
        for overlap in self.values["chunk_overlap"]:
            if overlap < fixed["chunk_size"]:
                valid_overlaps.append(overlap)

        if not valid_overlaps:
            raise ValueError(
                f"Kein chunk_overlap-Wert kleiner als chunk_size "
                f"{fixed['chunk_size']} im Suchraum vorhanden."
            )
        fixed["chunk_overlap"] = random.choice(valid_overlaps)
        return fixed

    def is_valid(self, individual: dict[str, Any]) -> bool:
        """Checks the chunk constraint without repairing it.

        Used by grid search, where a repair would create duplicates and
        introduce randomness into the enumeration.
        """
        return individual["chunk_overlap"] < individual["chunk_size"]

    def mutate_gene(self, gene_name: str, current: Any) -> Any:
        """Mutates a single gene value.

        Int genes step one position in the sorted grid and step inwards at the edges,
        categorical genes pick a random different value.
        """
        vals = self.values[gene_name]
        if len(vals) == 1:
            return current

        if isinstance(current, int):
            idx = vals.index(current)
            if idx == 0:
                return vals[1]
            if idx == len(vals) - 1:
                return vals[idx - 1]
            return vals[idx + random.choice((-1, 1))]

        others = [v for v in vals if v != current]
        return random.choice(others)
