"""Evaluation and full-match diagnostic utilities."""

from src.evaluation.diagnostics import (
    STANDARD_SLICES,
    compare_models_full_match,
    diagnose_prediction,
    diagnose_slice,
)

__all__ = [
    "STANDARD_SLICES",
    "diagnose_slice",
    "diagnose_prediction",
    "compare_models_full_match",
]

