"""Data loading and hygiene utilities for tactical tracking and metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl

# Standard pitch dimensions and match specifications
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
FPS = 25.0

# Frame ranges for BAR-ATH
N_TOTAL_FRAMES = 199319
HALF1_START, HALF1_END = 11185, 90651
HALF2_START, HALF2_END = 110808, 190197
PUBLIC_WIN_START, PUBLIC_WIN_END = 11185, 18684

# The 3 ghost slots to exclude (from Q4.1 EDA: 0 graded frames / high duplicate ghost tracks)
EXCLUDED_SLOT_IDS = {22, 23, 44}


@dataclass
class MatchData:
    """Cleaned tracking arrays and match structure."""
    frames: np.ndarray             # (N,) frame indices 0..199318
    ball_x_norm: np.ndarray        # (N,) normalized [0, 1]
    ball_y_norm: np.ndarray        # (N,) normalized [0, 1]
    ball_present: np.ndarray       # (N,) bool: True where ball coordinates are not NaN
    px_norm: np.ndarray            # (N, 43) normalized [0, 1] for 43 valid slots
    py_norm: np.ndarray            # (N, 43) normalized [0, 1]
    slot_ids: np.ndarray           # (43,) original slot ids 1..46 excluding {22, 23, 44}
    slot_is_home: np.ndarray       # (43,) bool: True if slot is Home (Barcelona)
    slot_is_away: np.ndarray       # (43,) bool: True if slot is Away (Athletic Club)
    graded_mask: np.ndarray        # (N,) bool: True on graded frames (Half 1 + Half 2)
    win_mask: np.ndarray           # (N,) bool: True on 5-min public ground truth window
    ps_norm: Optional[np.ndarray] = None  # (N, 43) player speed in m/s for 43 valid slots


def get_default_data_dir() -> Path:
    """Return the default processed tactical data directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "tactical"


def load_match_data(data_dir: Optional[Path] = None) -> MatchData:
    """Load tracking and slot parquets with slot hygiene applied."""
    if data_dir is None:
        data_dir = get_default_data_dir()

    tracking_path = data_dir / "tracking.parquet"
    slots_path = data_dir / "player_slots.parquet"

    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking parquet not found at: {tracking_path}")
    if not slots_path.exists():
        raise FileNotFoundError(f"Slots parquet not found at: {slots_path}")

    tracking_df = pl.read_parquet(tracking_path)
    slots_df = pl.read_parquet(slots_path)

    # 1. Slot hygiene: filter slots to the 43 active ones
    valid_slots_df = slots_df.filter(~pl.col("slot_id").is_in(list(EXCLUDED_SLOT_IDS))).sort("slot_id")
    valid_slot_ids = valid_slots_df["slot_id"].to_numpy()
    is_home_mask = (valid_slots_df["team"] == "H").to_numpy()
    is_away_mask = ~is_home_mask

    # 2. Extract ball coordinates
    frames = tracking_df["frame"].to_numpy()
    ball_x_norm = tracking_df["ball_x"].to_numpy()
    ball_y_norm = tracking_df["ball_y"].to_numpy()
    ball_present = ~np.isnan(ball_x_norm) & ~np.isnan(ball_y_norm)

    # 3. Extract player coordinates and speeds only for valid slots (shape N x 43)
    px_cols = [f"p{sid}_x" for sid in valid_slot_ids]
    py_cols = [f"p{sid}_y" for sid in valid_slot_ids]
    ps_cols = [f"p{sid}_speed" for sid in valid_slot_ids]
    px_norm = np.column_stack([tracking_df[col].to_numpy() for col in px_cols])
    py_norm = np.column_stack([tracking_df[col].to_numpy() for col in py_cols])
    ps_norm = (
        np.column_stack([tracking_df[col].to_numpy() for col in ps_cols])
        if all(col in tracking_df.columns for col in ps_cols)
        else None
    )

    # 4. Construct masks
    n_frames = len(frames)
    graded_mask = np.zeros(n_frames, dtype=bool)
    graded_mask[HALF1_START : HALF1_END + 1] = True
    graded_mask[HALF2_START : HALF2_END + 1] = True

    win_mask = np.zeros(n_frames, dtype=bool)
    win_mask[PUBLIC_WIN_START : PUBLIC_WIN_END + 1] = True

    return MatchData(
        frames=frames,
        ball_x_norm=ball_x_norm,
        ball_y_norm=ball_y_norm,
        ball_present=ball_present,
        px_norm=px_norm,
        py_norm=py_norm,
        slot_ids=valid_slot_ids,
        slot_is_home=is_home_mask,
        slot_is_away=is_away_mask,
        graded_mask=graded_mask,
        win_mask=win_mask,
        ps_norm=ps_norm,
    )


def load_smart_tagging(data_dir: Optional[Path] = None) -> pl.DataFrame:
    """Load processed Smart Tagging interval events parquet."""
    if data_dir is None:
        data_dir = get_default_data_dir()

    tagging_path = data_dir / "smart_tagging.parquet"
    if not tagging_path.exists():
        raise FileNotFoundError(f"Smart tagging parquet not found at: {tagging_path}")
    return pl.read_parquet(tagging_path)


def compute_team_distances_to_ball(data: MatchData) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distance in metres from ball to nearest home and away player.

    Returns:
        (d_H, d_A): arrays of shape (N,) with distance in metres.
        Values are NaN on frames where ball is absent.
    """
    # Vectorized displacement in metres: (N, 43)
    dx_m = (data.px_norm - data.ball_x_norm[:, None]) * PITCH_LENGTH
    dy_m = (data.py_norm - data.ball_y_norm[:, None]) * PITCH_WIDTH
    dist_m = np.hypot(dx_m, dy_m)

    # Mask distances by team
    dist_home_m = np.where(data.slot_is_home[None, :], dist_m, np.inf)
    dist_away_m = np.where(data.slot_is_away[None, :], dist_m, np.inf)

    with np.errstate(all="ignore"):
        d_H = np.nanmin(np.where(np.isnan(dist_home_m), np.inf, dist_home_m), axis=1)
        d_A = np.nanmin(np.where(np.isnan(dist_away_m), np.inf, dist_away_m), axis=1)

    # Set NaN when ball is not present
    d_H[~data.ball_present] = np.nan
    d_A[~data.ball_present] = np.nan

    return d_H, d_A

