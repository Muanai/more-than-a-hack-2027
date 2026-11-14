"""Feature extraction package for physical and occlusion-aware possession signals."""

from src.features.possession_features import (
    DEFENSIVE_CODE_WEIGHTS,
    HALF_RANGES,
    OFFENSIVE_CODE_WEIGHTS,
    PossessionFeatures,
    box_smooth_half,
    build_possession_features,
    compute_smart_tagging_signals,
    forward_backfill_mask,
    prune_binary_runs,
)

__all__ = [
    "HALF_RANGES",
    "OFFENSIVE_CODE_WEIGHTS",
    "DEFENSIVE_CODE_WEIGHTS",
    "PossessionFeatures",
    "box_smooth_half",
    "prune_binary_runs",
    "forward_backfill_mask",
    "compute_smart_tagging_signals",
    "build_possession_features",
]

