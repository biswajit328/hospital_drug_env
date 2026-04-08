"""Shared score normalization helpers for validator-safe task scoring."""

from __future__ import annotations

MIN_VALID_SCORE = 0.01
MAX_VALID_SCORE = 0.99


def clamp_validator_safe_score(score: float | None) -> float | None:
    """Force externally reported task scores into the strict (0, 1) interval."""
    if score is None:
        return None

    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Score out of normalized range [0, 1]: {value}")

    return round(min(MAX_VALID_SCORE, max(MIN_VALID_SCORE, value)), 3)
