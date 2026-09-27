"""Aggregation of individual metrics into a weighted composite score."""

from __future__ import annotations


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Aggregates the individual metrics into the weighted score the GA maximizes.

    Raises:
        ValueError: If a key from ``weights`` is missing in ``metrics`` or the
            weights do not sum to ``1.0``.
    """
    missing = [key for key in weights if key not in metrics]
    if missing:
        raise ValueError(
            f"Folgende Metrik-Keys werden von weights benötigt, fehlen aber in metrics: {missing}"
        )

    total = sum(weights.values())
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"weights muss in Summe 1.0 ergeben, aktuell {total:.6f}.")

    return sum(metrics[key] * weight for key, weight in weights.items())
