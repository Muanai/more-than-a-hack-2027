"""Baseline nearest-player possession pipelines.

Implements the official distance-to-ball attribution logic with slot hygiene,
vectorized smoothing, evaluation against ground truth, and submission export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from src.data.loader import MatchData, compute_team_distances_to_ball, load_match_data
from src.evaluation.diagnostics import diagnose_prediction
from src.metrics import possession_metrics as pm

# Internal team codes for baseline attribution
HOME = 0
AWAY = 1
NOBODY = 2


@dataclass
class BaselineResult:
    """Output container for baseline predictions and evaluation."""
    labels: np.ndarray                     # (N,) string array: 'H', 'A', 'D'
    raw_codes: np.ndarray                  # (N,) int8 array: HOME (0), AWAY (1), NOBODY (2)
    metrics: Optional[Dict] = None         # Metric evaluation dict if ground truth available
    diagnostics: Optional[pl.DataFrame] = None  # Label-free full-match structural diagnostics
    window: int = 0                        # Smoothing half-window size in frames


def predict_nearest_player_raw(data: MatchData) -> np.ndarray:
    """Assign ball to nearest team per-frame based on Euclidean distance in metres.

    Returns:
        raw_codes: shape (N,) int8 array with HOME (0), AWAY (1), or NOBODY (2).
    """
    d_H, d_A = compute_team_distances_to_ball(data)

    raw_codes = np.full(len(data.frames), NOBODY, dtype=np.int8)
    alive = data.ball_present
    raw_codes[alive & (d_H < d_A)] = HOME
    raw_codes[alive & (d_H >= d_A)] = AWAY
    return raw_codes


def apply_temporal_smoothing(raw_codes: np.ndarray, window: int = 0) -> np.ndarray:
    """Apply centered rolling majority vote over +/- window frames.

    Args:
        raw_codes: shape (N,) array of HOME (0), AWAY (1), NOBODY (2)
        window: half-window in frames (0 = raw / no smoothing)

    Returns:
        labels: shape (N,) string array with 'H', 'A', or 'D'
    """
    if window <= 0:
        return np.where(raw_codes == HOME, "H", np.where(raw_codes == AWAY, "A", "D"))

    # Vectorized cumulative sum implementation matching official benchmark baseline
    home_cum = np.concatenate(([0], np.cumsum(raw_codes == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(raw_codes == AWAY)))

    idx = np.arange(len(raw_codes))
    low = np.maximum(0, idx - window)
    high = np.minimum(len(raw_codes), idx + window + 1)

    home_counts = home_cum[high] - home_cum[low]
    away_counts = away_cum[high] - away_cum[low]

    labels = np.full(len(raw_codes), "D", dtype="<U1")
    seen = (home_counts + away_counts) > 0
    # Ties break to Home (FIFA/EPTS convention)
    labels[seen] = np.where(home_counts[seen] >= away_counts[seen], "H", "A")
    return labels


def run_baseline_pipeline(
    data: Optional[MatchData] = None,
    window: int = 0,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Run end-to-end baseline pipeline with given smoothing window.

    Args:
        data: MatchData instance; loads default if None
        window: smoothing half-width in frames (e.g. 0, 25, 50, 125)
        gt_csv_path: path to public ground truth CSV for evaluation

    Returns:
        BaselineResult with predicted labels, raw codes, and metrics
    """
    if data is None:
        data = load_match_data()

    raw_codes = predict_nearest_player_raw(data)
    labels = apply_temporal_smoothing(raw_codes, window=window)

    metrics = None
    if gt_csv_path is None:
        default_gt = Path(__file__).resolve().parent.parent.parent / "reference" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
        if default_gt.exists():
            gt_csv_path = default_gt

    if gt_csv_path is not None and Path(gt_csv_path).exists():
        gt_df = pm.read_label_csv(str(gt_csv_path))
        gt_codes = pm.label_codes(gt_df)
        pred_codes = pm.label_codes(labels)
        metrics = pm.evaluate(gt_codes, pred_codes)

    diagnostics = diagnose_prediction(labels, model_name=f"baseline_w{window}")

    return BaselineResult(
        labels=labels,
        raw_codes=raw_codes,
        metrics=metrics,
        diagnostics=diagnostics,
        window=window,
    )


def save_submission(
    labels: np.ndarray,
    out_path: Union[str, Path],
    ungraded_to_x: bool = False,
    graded_mask: Optional[np.ndarray] = None,
) -> Path:
    """Export predictions to a submission CSV compliant with validate_submission.

    Args:
        labels: (N,) string labels
        out_path: destination CSV path
        ungraded_to_x: if True, sets ungraded frames to 'X' (strictly optional, free)
        graded_mask: bool mask for graded frames (required if ungraded_to_x is True)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_labels = labels.copy()
    if ungraded_to_x and graded_mask is not None:
        export_labels[~graded_mask] = "X"

    sub_df = pd.DataFrame({
        "frame": np.arange(len(export_labels)),
        "label": export_labels,
    })

    # Validate before writing
    reason = pm.validate_submission(sub_df, len(export_labels))
    if reason is not None:
        raise ValueError(f"Submission validation failed: {reason}")

    sub_df.to_csv(out_path, index=False)
    return out_path

