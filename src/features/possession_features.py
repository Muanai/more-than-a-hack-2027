"""Physics-informed and occlusion-aware frame feature extraction for ball possession."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.data.loader import (
    FPS,
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    PITCH_LENGTH,
    PITCH_WIDTH,
    MatchData,
    compute_team_distances_to_ball,
    load_match_data,
    load_smart_tagging,
)

HALF_RANGES: Tuple[Tuple[int, int], ...] = (
    (HALF1_START, HALF1_END + 1),
    (HALF2_START, HALF2_END + 1),
)

# Implied possessor weights from Q7.3 (>=60% alive-frame agreement; drop spatial <60% codes)
OFFENSIVE_CODE_WEIGHTS: Dict[str, float] = {
    "BUILD UP": 1.0,
    "CREATING CHANCES": 1.0,
    "SET PIECES": 0.95,
    "LONG BALL": 0.9,
    "PROGRESSION": 0.65,
    "ATTACKING TRANSITION": 0.55,
}
DEFENSIVE_CODE_WEIGHTS: Dict[str, float] = {
    "DEFENDING IN ATTACKING THIRD": 1.0,
    "DEFENDING IN DEFENSIVE THIRD": 1.0,
    "DEFENDING IN MIDDLE THIRD": 0.65,
    "DEFENSIVE TRANSITION": 0.55,
}


@dataclass
class PossessionFeatures:
    """Aligned (N,) frame-level feature arrays partitioned by physical domain."""

    features: Dict[str, np.ndarray]
    feature_names: List[str]
    d_mask: np.ndarray
    nw_margin: np.ndarray
    den_nw: np.ndarray
    tag_prior_raw: np.ndarray
    tag_prior_dedup: np.ndarray
    sp_act: np.ndarray

    def to_matrix(self, names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """Stack selected feature columns into a float64 matrix of shape (N, F)."""
        cols = self.feature_names if names is None else names
        mat = np.column_stack([self.features[name].astype(np.float64, copy=False) for name in cols])
        return mat, cols

    def to_polars(self, names: Optional[List[str]] = None) -> pl.DataFrame:
        """Convert selected features into a Polars DataFrame."""
        cols = self.feature_names if names is None else names
        return pl.DataFrame({name: self.features[name] for name in cols})


def box_smooth_half(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute centered rolling mean inside each graded half without bridging halftime."""
    out = np.zeros(len(arr), dtype=np.float64)
    if window <= 0:
        for start, stop in HALF_RANGES:
            out[start:stop] = arr[start:stop]
        return out

    for start, stop in HALF_RANGES:
        sub = np.asarray(arr[start:stop], dtype=np.float64)
        n = len(sub)
        cs = np.concatenate(([0.0], np.cumsum(sub)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - window)
        hi = np.minimum(n, idx + window + 1)
        out[start:stop] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def prune_binary_runs(mask_arr: np.ndarray, min_true: int, min_false: int) -> np.ndarray:
    """Apply morphological opening-before-closing inside each half to remove blips and micro-gaps."""
    arr = mask_arr.copy()
    opening_len = min(40, min_true)
    passes = ((True, opening_len), (False, min_false), (True, min_true))

    for start, stop in HALF_RANGES:
        sub = arr[start:stop]
        for target_val, min_len in passes:
            if min_len <= 1 or len(sub) == 0:
                continue
            edges = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.concatenate(([0], edges))
            stops = np.concatenate((edges, [len(sub)]))
            for a, b in zip(starts, stops):
                if sub[a] == target_val and (b - a) < min_len:
                    sub[a:b] = not target_val
        arr[start:stop] = sub
    return arr


def forward_backfill_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Carry last valid observation forward, backfilling any leading invalid prefix."""
    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) == 0:
        return np.zeros_like(values)

    carry = np.where(valid_mask, np.arange(len(values)), -1)
    np.maximum.accumulate(carry, out=carry)
    carry[carry < 0] = valid_idx[0]
    return values[carry]


def compute_smart_tagging_signals(
    n_frames: int,
    tag_df: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute summed prior, non-double-counted polarity prior, and SET PIECES flag."""
    tag_raw = np.zeros(n_frames, dtype=np.float64)
    tag_off = np.zeros(n_frames, dtype=np.float64)
    tag_def = np.zeros(n_frames, dtype=np.float64)
    sp_act = np.zeros(n_frames, dtype=np.float64)

    for row in tag_df.iter_rows(named=True):
        s = max(0, int(row["start_frame"]))
        e = min(n_frames, int(row["end_frame"]))
        if s >= e:
            continue
        code = row["code"]
        team = row["team"]
        if code == "SET PIECES":
            sp_act[s:e] = 1.0
        if team not in ("Barcelona", "Atletic Club"):
            continue
        sgn = 1.0 if team == "Barcelona" else -1.0
        if code in OFFENSIVE_CODE_WEIGHTS:
            val = sgn * OFFENSIVE_CODE_WEIGHTS[code]
            tag_raw[s:e] += val
            upd = np.abs(val) > np.abs(tag_off[s:e])
            tag_off[s:e] = np.where(upd, val, tag_off[s:e])
        elif code in DEFENSIVE_CODE_WEIGHTS:
            val = -sgn * DEFENSIVE_CODE_WEIGHTS[code]
            tag_raw[s:e] += val
            upd = np.abs(val) > np.abs(tag_def[s:e])
            tag_def[s:e] = np.where(upd, val, tag_def[s:e])

    tag_dedup = np.where(tag_off != 0.0, tag_off, tag_def)
    return np.clip(tag_raw, -1.5, 1.5), tag_dedup, sp_act


def _safe_masked_mean(values: np.ndarray, mask: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Compute row-wise mean without raising RuntimeWarning on all-NaN slices."""
    valid = mask & ~np.isnan(values)
    cnt = np.sum(valid, axis=1)
    total = np.sum(np.where(valid, values, 0.0), axis=1)
    return np.where(cnt > 0, total / np.maximum(cnt, 1), default)


def _compute_oob_causal(
    bx: np.ndarray,
    by: np.ndarray,
    bs_s: np.ndarray,
    ball_present: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw OOB flag and causal OOB/goal-wait stoppage hold per half."""
    n_frames = len(bx)
    oob = ball_present & ((bx < 0.0) | (bx > 1.0) | (by < 0.0) | (by > 1.0))
    in_goal_wait = (
        ball_present
        & ((bx < 0.045) | (bx > 0.955))
        & (by > 0.25)
        & (by < 0.75)
        & (bs_s < 4.5)
    )

    oob_causal = np.zeros(n_frames, dtype=bool)
    for start, stop in HALF_RANGES:
        hold = 0
        for idx in range(start, stop):
            if oob[idx]:
                hold = 90
                oob_causal[idx] = True
            elif hold > 0:
                if (not oob[idx]) and (bs_s[idx] > 5.5) and ball_present[idx]:
                    hold = 0
                elif in_goal_wait[idx]:
                    oob_causal[idx] = True
                else:
                    oob_causal[idx] = True
                    hold -= 1
    return oob, oob_causal


def _compute_kick_impulse(bvx: np.ndarray, bvy: np.ndarray, valid_ball: np.ndarray, k: int = 4) -> np.ndarray:
    """Measure pre/post k-frame vector velocity change of the ball (m/s) within each half."""
    n = len(bvx)
    impulse = np.zeros(n, dtype=np.float64)
    vb = (valid_ball & np.isfinite(bvx) & np.isfinite(bvy)).astype(np.float64)

    for start, stop in HALF_RANGES:
        m = stop - start
        if m <= 2 * k:
            continue
        csx = np.concatenate(([0.0], np.cumsum(np.where(vb[start:stop] > 0, bvx[start:stop], 0.0))))
        csy = np.concatenate(([0.0], np.cumsum(np.where(vb[start:stop] > 0, bvy[start:stop], 0.0))))
        csc = np.concatenate(([0.0], np.cumsum(vb[start:stop])))
        idx = np.arange(m)

        lo_pre = np.maximum(0, idx - k)
        hi_pre = idx
        lo_post = np.minimum(m, idx + 1)
        hi_post = np.minimum(m, idx + k + 1)

        c_pre = csc[hi_pre] - csc[lo_pre]
        c_post = csc[hi_post] - csc[lo_post]
        vx_pre = np.where(c_pre > 0, (csx[hi_pre] - csx[lo_pre]) / np.maximum(c_pre, 1.0), 0.0)
        vy_pre = np.where(c_pre > 0, (csy[hi_pre] - csy[lo_pre]) / np.maximum(c_pre, 1.0), 0.0)
        vx_post = np.where(c_post > 0, (csx[hi_post] - csx[lo_post]) / np.maximum(c_post, 1.0), 0.0)
        vy_post = np.where(c_post > 0, (csy[hi_post] - csy[lo_post]) / np.maximum(c_post, 1.0), 0.0)

        has_both = (c_pre >= 2) & (c_post >= 2)
        impulse[start:stop] = np.where(has_both, np.hypot(vx_post - vx_pre, vy_post - vy_pre), 0.0)
    return impulse


def build_possession_features(
    data: Optional[MatchData] = None,
    tag_df: Optional[pl.DataFrame] = None,
) -> PossessionFeatures:
    """Build the full Step 1 physics, ball-control, and occlusion-aware feature set."""
    if data is None:
        data = load_match_data()
    if tag_df is None:
        tag_df = load_smart_tagging()
    if data.ps_norm is None:
        raise ValueError("MatchData.ps_norm is required; reload with load_match_data().")

    n = len(data.frames)
    bx, by = data.ball_x_norm, data.ball_y_norm
    ball_present = data.ball_present

    # --- 1. Ball kinematics & jump gate ---
    bvx = np.full(n, np.nan, dtype=np.float64)
    bvy = np.full(n, np.nan, dtype=np.float64)
    for start, stop in HALF_RANGES:
        consec = ball_present[start + 1 : stop] & ball_present[start : stop - 1]
        bvx[start + 1 : stop][consec] = np.diff(bx[start:stop])[consec] * PITCH_LENGTH * FPS
        bvy[start + 1 : stop][consec] = np.diff(by[start:stop])[consec] * PITCH_WIDTH * FPS
    bs = np.hypot(bvx, bvy)
    valid_ball = ball_present & (np.isnan(bs) | (bs <= 40.0))
    bs_clean = np.where(np.isnan(bs) | (bs > 40.0), 5.0, bs)
    bs_s = box_smooth_half(bs_clean, 25)
    bs_fast = box_smooth_half(bs_clean, 6)
    kick_impulse = _compute_kick_impulse(bvx, bvy, valid_ball, k=4)

    # --- 2. Collective player tempo, slowdown, walking fraction & p90 ---
    ps = data.ps_norm
    valid_ps = ~np.isnan(ps)
    ps_cnt = np.sum(valid_ps, axis=1)
    mean_ps = _safe_masked_mean(ps, np.ones_like(ps, dtype=bool), default=1.8)
    tempo_mean = box_smooth_half(mean_ps, 3750)
    slowdown_diff = box_smooth_half(mean_ps, 35) - tempo_mean
    sd_fast = box_smooth_half(mean_ps, 12) - tempo_mean

    frac_slow = np.where(
        ps_cnt > 0,
        np.sum(np.where(valid_ps, ps < 1.5, False), axis=1) / np.maximum(ps_cnt, 1),
        0.4,
    )
    frac_slow_diff = box_smooth_half(frac_slow, 18) - box_smooth_half(frac_slow, 3750)

    ps_sorted = np.sort(np.where(valid_ps, ps, -1.0), axis=1)
    p90_idx = np.clip(ps_sorted.shape[1] - np.maximum(1, np.round(0.10 * ps_cnt).astype(int)), 0, ps_sorted.shape[1] - 1)
    ps_p90 = np.where(ps_cnt > 0, ps_sorted[np.arange(n), p90_idx], 3.5)
    p90_diff = box_smooth_half(ps_p90, 18) - box_smooth_half(ps_p90, 3750)

    bp_s = box_smooth_half(ball_present.astype(np.float64), 45)
    oob, oob_causal = _compute_oob_causal(bx, by, bs_s, ball_present)

    # --- 3. Player-ball proximity, 2nd nearest, local support & velocity alignment ---
    d_H, d_A = compute_team_distances_to_ball(data)
    d_min = np.minimum(d_H, d_A)
    dx_m = (data.px_norm - bx[:, None]) * PITCH_LENGTH
    dy_m = (data.py_norm - by[:, None]) * PITCH_WIDTH
    dist_m = np.hypot(dx_m, dy_m)

    d_h_mat = np.where(data.slot_is_home[None, :] & ~np.isnan(dist_m), dist_m, np.inf)
    d_a_mat = np.where(data.slot_is_away[None, :] & ~np.isnan(dist_m), dist_m, np.inf)
    d_h2 = np.partition(d_h_mat, 1, axis=1)[:, 1]
    d_a2 = np.partition(d_a_mat, 1, axis=1)[:, 1]
    d_h2 = np.where(ball_present & np.isfinite(d_h2), d_h2, 25.0)
    d_a2 = np.where(ball_present & np.isfinite(d_a2), d_a2, 25.0)

    cnt_h_3m = np.where(ball_present, np.sum(d_h_mat < 3.0, axis=1), 0).astype(np.float64)
    cnt_a_3m = np.where(ball_present, np.sum(d_a_mat < 3.0, axis=1), 0).astype(np.float64)
    support_margin = np.where(valid_ball, np.tanh((cnt_h_3m - cnt_a_3m) / 1.5), 0.0)

    pvx_raw = np.zeros_like(data.px_norm)
    pvy_raw = np.zeros_like(data.py_norm)
    pvx_valid = np.zeros(data.px_norm.shape, dtype=bool)
    for start, stop in HALF_RANGES:
        dx_p = np.diff(data.px_norm[start:stop], axis=0) * PITCH_LENGTH * FPS
        dy_p = np.diff(data.py_norm[start:stop], axis=0) * PITCH_WIDTH * FPS
        consec_p = ~np.isnan(dx_p) & ~np.isnan(dy_p)
        pvx_raw[start + 1 : stop] = np.where(consec_p, dx_p, 0.0)
        pvy_raw[start + 1 : stop] = np.where(consec_p, dy_p, 0.0)
        pvx_valid[start + 1 : stop] = consec_p

    rows = np.arange(n)
    idx_h = np.argmin(d_h_mat, axis=1)
    idx_a = np.argmin(d_a_mat, axis=1)
    bvx_0 = np.where(np.isfinite(bvx), bvx, 0.0)
    bvy_0 = np.where(np.isfinite(bvy), bvy, 0.0)
    rel_v_H = np.where(valid_ball, np.hypot(pvx_raw[rows, idx_h] - bvx_0, pvy_raw[rows, idx_h] - bvy_0), 0.0)
    rel_v_A = np.where(valid_ball, np.hypot(pvx_raw[rows, idx_a] - bvx_0, pvy_raw[rows, idx_a] - bvy_0), 0.0)
    rel_v_margin = np.where(valid_ball, np.tanh((rel_v_A - rel_v_H) / 3.5), 0.0)

    # Zero-phase Nadaraya-Watson ball-control smoothing
    raw_margin = np.zeros(n, dtype=np.float64)
    raw_margin[valid_ball] = np.tanh((d_A[valid_ball] - d_H[valid_ball]) / 1.5)
    d_margin2 = np.where(valid_ball, np.tanh((d_a2 - d_h2) / 2.5), 0.0)

    w_dist = np.exp(-np.maximum(0.0, np.where(valid_ball, d_min, 10.0) - 1.5) / 2.5)
    w_speed = np.exp(-np.maximum(0.0, bs_clean - 8.0) / 8.0)
    w_ctrl = np.where(valid_ball, np.clip(w_dist * w_speed, 0.05, 1.0), 0.0)

    den_nw = box_smooth_half(w_ctrl, 38)
    valid_nw = den_nw > 1e-4

    def _nw_smooth(sig: np.ndarray) -> np.ndarray:
        num = box_smooth_half(w_ctrl * sig, 38)
        m = np.where(valid_nw, num / np.maximum(den_nw, 1e-4), 0.0)
        return forward_backfill_mask(m, valid_nw)

    nw_margin = _nw_smooth(raw_margin)
    nw_support = _nw_smooth(support_margin)
    nw_rel_v = _nw_smooth(rel_v_margin)

    # --- 4. Occlusion-aware collective team motion & spatial shape (Halftime-normalized) ---
    px_att = data.px_norm.copy()
    px_att[HALF2_START : HALF2_END + 1] = 1.0 - px_att[HALF2_START : HALF2_END + 1]
    pvx_att = pvx_raw.copy()
    pvx_att[HALF2_START : HALF2_END + 1] = -pvx_att[HALF2_START : HALF2_END + 1]

    all_vx_att = np.clip(_safe_masked_mean(pvx_att, pvx_valid, default=0.0), -5.0, 5.0)
    h_vx_att = np.clip(_safe_masked_mean(pvx_att, pvx_valid & data.slot_is_home[None, :], default=0.0), -5.0, 5.0)
    a_vx_att = np.clip(_safe_masked_mean(pvx_att, pvx_valid & data.slot_is_away[None, :], default=0.0), -5.0, 5.0)

    all_vx_s = box_smooth_half(all_vx_att, 30)
    h_vx_s = box_smooth_half(h_vx_att, 30)
    a_vx_s = box_smooth_half(a_vx_att, 30)

    h_mean_x_att = _safe_masked_mean(px_att, data.slot_is_home[None, :], default=0.5)
    a_mean_x_att = _safe_masked_mean(px_att, data.slot_is_away[None, :], default=0.5)
    pitch_tilt = 0.5 * (h_mean_x_att + a_mean_x_att) - 0.5

    # --- 5. Smart Tagging & Pitch Context + Decoupled D Mask ---
    tag_raw, tag_dedup, sp_act = compute_smart_tagging_signals(n, tag_df)

    uncontrolled_slow = ball_present & (d_min > 6.0) & (slowdown_diff < -0.25)
    ball_not_fast = (~ball_present) | oob | (bs_s < 5.2) | (d_min > 4.5)
    d_cand1 = (
        ((slowdown_diff < -0.78) & ball_not_fast)
        | ((bp_s < 0.25) & (slowdown_diff < -0.62))
        | oob_causal
        | uncontrolled_slow
    )
    d_cand2 = d_cand1 | ((sp_act > 0.0) & (slowdown_diff < -0.62) & (bs_s < 3.2))
    d_mask = prune_binary_runs(d_cand2, min_true=115, min_false=65)

    bx_att = np.where(ball_present, bx, 0.5)
    bx_att[HALF2_START : HALF2_END + 1] = 1.0 - bx_att[HALF2_START : HALF2_END + 1]
    by_filled = np.where(ball_present, by, 0.5)
    bx_edge_dist = np.minimum(bx_att, 1.0 - bx_att)
    by_edge_dist = np.minimum(by_filled, 1.0 - by_filled)

    features: Dict[str, np.ndarray] = {
        # Dead-ball & collective tempo
        "slowdown_diff": slowdown_diff,
        "sd_fast": sd_fast,
        "frac_slow_diff": frac_slow_diff,
        "p90_diff": p90_diff,
        "bp_s": bp_s,
        "oob": oob.astype(np.float64),
        "oob_causal": oob_causal.astype(np.float64),
        "uncontrolled_slow": uncontrolled_slow.astype(np.float64),
        "ball_not_fast": ball_not_fast.astype(np.float64),
        "d_mask": d_mask.astype(np.float64),
        # Ball control, support, velocity alignment & touch impulse
        "valid_ball": valid_ball.astype(np.float64),
        "d_H_filled": np.where(valid_ball & np.isfinite(d_H), np.clip(d_H, 0.0, 50.0), 15.0),
        "d_A_filled": np.where(valid_ball & np.isfinite(d_A), np.clip(d_A, 0.0, 50.0), 15.0),
        "d_min_filled": np.where(valid_ball & np.isfinite(d_min), np.clip(d_min, 0.0, 50.0), 15.0),
        "d_margin_raw": raw_margin,
        "d_margin2": d_margin2,
        "support_diff_3m": cnt_h_3m - cnt_a_3m,
        "bs_clean": bs_clean,
        "bs_s": bs_s,
        "bs_fast": bs_fast,
        "kick_impulse": kick_impulse,
        "rel_v_margin": rel_v_margin,
        "w_ctrl": w_ctrl,
        "den_nw": den_nw,
        "nw_margin": nw_margin,
        "nw_support": nw_support,
        "nw_rel_v": nw_rel_v,
        # Occlusion-aware team kinematics
        "all_vx_s": all_vx_s,
        "h_vx_s": h_vx_s,
        "a_vx_s": a_vx_s,
        "h_mean_x_att": h_mean_x_att,
        "a_mean_x_att": a_mean_x_att,
        "pitch_tilt": pitch_tilt,
        # Smart Tagging & context
        "tag_prior_raw": tag_raw,
        "tag_prior_dedup": tag_dedup,
        "sp_act": sp_act,
        "bx_att": bx_att,
        "by_filled": by_filled,
        "bx_edge_dist": bx_edge_dist,
        "by_edge_dist": by_edge_dist,
    }

    return PossessionFeatures(
        features=features,
        feature_names=list(features.keys()),
        d_mask=d_mask,
        nw_margin=nw_margin,
        den_nw=den_nw,
        tag_prior_raw=tag_raw,
        tag_prior_dedup=tag_dedup,
        sp_act=sp_act,
    )
