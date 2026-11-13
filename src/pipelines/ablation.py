"""3-stage ablation study pipelines for ball possession segmentation.

Progressively evaluates:
1. Ablation 1 (Decoupled): tempo-normalized dead-ball detection + H/A carry-forward & min-dwell.
2. Ablation 2 (Features & Tagging): continuous team-ball margins, ball jump gate, and
   sign-corrected Smart Tagging priors (+1 offensive, -1 defensive, drop <60% spatial tags).
3. Ablation 3 (Sequence Decoder): 3-state (H, A, D) Viterbi DP decoder run separately per half.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
from src.evaluation.diagnostics import compare_models_full_match, diagnose_prediction
from src.metrics import possession_metrics as pm
from src.pipelines.baseline import (
    AWAY,
    HOME,
    NOBODY,
    BaselineResult,
    run_baseline_pipeline,
    save_submission,
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


def cleanup_short_spells(labels: np.ndarray, min_spell_frames: int = 25) -> np.ndarray:
    """Absorb sub-threshold H/A possession fragments inside each half into neighboring states."""
    out = labels.copy()
    for start, stop in HALF_RANGES:
        sub = out[start:stop]
        for _ in range(3):
            edges = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.concatenate(([0], edges))
            stops = np.concatenate((edges, [len(sub)]))
            changed = False
            for idx_seg, (a, b) in enumerate(zip(starts, stops)):
                if sub[a] in ("H", "A") and (b - a) < min_spell_frames:
                    prev_lbl = sub[starts[idx_seg - 1]] if idx_seg > 0 else None
                    next_lbl = sub[starts[idx_seg + 1]] if idx_seg + 1 < len(starts) else None
                    rep = prev_lbl if prev_lbl is not None else (next_lbl if next_lbl is not None else sub[a])
                    sub[a:b] = rep
                    changed = True
            if not changed:
                break
        out[start:stop] = sub
    return out


def _forward_backfill_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Carry last valid observation forward, backfilling any leading invalid prefix."""
    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) == 0:
        return np.zeros_like(values)

    carry = np.where(valid_mask, np.arange(len(values)), -1)
    np.maximum.accumulate(carry, out=carry)
    carry[carry < 0] = valid_idx[0]
    return values[carry]


def compute_kinematic_signals(data: MatchData) -> Dict[str, np.ndarray]:
    """Compute ball kinematics, team distances, tempo-normalized player speed, and OOB holds."""
    if data.ps_norm is None:
        raise ValueError("MatchData.ps_norm is required for ablation pipelines; reload with load_match_data().")

    d_H, d_A = compute_team_distances_to_ball(data)
    d_min = np.minimum(d_H, d_A)

    bx = data.ball_x_norm
    by = data.ball_y_norm
    n_frames = len(bx)

    # Finite-difference ball speed in m/s (ignoring tracker >40 m/s jumps)
    consec_bp = data.ball_present[1:] & data.ball_present[:-1]
    bs = np.full(n_frames, np.nan, dtype=np.float64)
    dx_m = np.diff(bx) * PITCH_LENGTH
    dy_m = np.diff(by) * PITCH_WIDTH
    bs[1:][consec_bp] = np.hypot(dx_m[consec_bp], dy_m[consec_bp]) * FPS

    valid_ball = data.ball_present & (np.isnan(bs) | (bs <= 40.0))
    bs_clean = np.where(np.isnan(bs) | (bs > 40.0), 5.0, bs)
    bs_s = box_smooth_half(bs_clean, 25)

    # Collective player speed & additive tempo adjustment per half (avoiding empty-slice warnings)
    ps_valid_cnt = np.sum(~np.isnan(data.ps_norm), axis=1)
    ps_sum = np.nansum(data.ps_norm, axis=1)
    mean_ps = np.where(ps_valid_cnt > 0, ps_sum / np.maximum(ps_valid_cnt, 1), 1.8)
    mean_ps_s = box_smooth_half(mean_ps, 35)
    tempo_mean = box_smooth_half(mean_ps, 3750)
    slowdown_diff = mean_ps_s - tempo_mean

    bp_s = box_smooth_half(data.ball_present.astype(np.float64), 45)
    oob = data.ball_present & ((bx < 0.0) | (bx > 1.0) | (by < 0.0) | (by > 1.0))
    in_goal_wait = (
        data.ball_present
        & ((bx < 0.045) | (bx > 0.955))
        & (by > 0.25)
        & (by < 0.75)
        & (bs_s < 4.5)
    )

    # Causal OOB hold prevents symmetric smoothing from pre-smearing D before OOB or after restarts
    oob_causal = np.zeros(n_frames, dtype=bool)
    for start, stop in HALF_RANGES:
        hold = 0
        for idx in range(start, stop):
            if oob[idx]:
                hold = 90
                oob_causal[idx] = True
            elif hold > 0:
                if (not oob[idx]) and (bs_s[idx] > 5.5) and data.ball_present[idx]:
                    hold = 0
                elif in_goal_wait[idx]:
                    oob_causal[idx] = True
                else:
                    oob_causal[idx] = True
                    hold -= 1

    uncontrolled_slow = data.ball_present & (d_min > 6.0) & (slowdown_diff < -0.25)
    ball_not_fast = (~data.ball_present) | oob | (bs_s < 5.2) | (d_min > 4.5)

    # Attack-direction-normalized ball longitudinal coordinate (Home always attacks Left -> Right)
    bx_att = bx.copy()
    bx_att[HALF2_START : HALF2_END + 1] = 1.0 - bx_att[HALF2_START : HALF2_END + 1]

    return {
        "d_H": d_H,
        "d_A": d_A,
        "d_min": d_min,
        "bs": bs,
        "bs_clean": bs_clean,
        "bs_s": bs_s,
        "valid_ball": valid_ball,
        "mean_ps": mean_ps,
        "slowdown_diff": slowdown_diff,
        "bp_s": bp_s,
        "oob_causal": oob_causal,
        "uncontrolled_slow": uncontrolled_slow,
        "ball_not_fast": ball_not_fast,
        "bx_att": bx_att,
    }


def compute_smart_tagging_priors(
    n_frames: int,
    tag_df: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sign-corrected frame-level Smart Tagging priors and SET PIECES indicator."""
    tag_prior = np.zeros(n_frames, dtype=np.float64)
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
            tag_prior[s:e] += sgn * OFFENSIVE_CODE_WEIGHTS[code]
        elif code in DEFENSIVE_CODE_WEIGHTS:
            tag_prior[s:e] -= sgn * DEFENSIVE_CODE_WEIGHTS[code]

    return np.clip(tag_prior, -1.5, 1.5), sp_act


def viterbi_decode_half(emit_scores: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    """Decode optimal 3-state path over a single continuous half using Viterbi DP."""
    t_len, n_states = emit_scores.shape
    dp = np.zeros((t_len, n_states), dtype=np.float64)
    back = np.zeros((t_len, n_states), dtype=np.int8)
    dp[0] = emit_scores[0]

    for t in range(1, t_len):
        scores = dp[t - 1][:, None] + trans_mat
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(n_states)] + emit_scores[t]
        back[t] = best_prev

    path = np.zeros(t_len, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _resolve_gt_path(gt_csv_path: Optional[Union[str, Path]]) -> Optional[Path]:
    if gt_csv_path is not None:
        p = Path(gt_csv_path)
        return p if p.exists() else None
    default_gt = (
        Path(__file__).resolve().parent.parent.parent
        / "reference"
        / "evaluation"
        / "gt"
        / "BAR-ATH_possession_gt_public.csv"
    )
    return default_gt if default_gt.exists() else None


def _finalize_result(
    labels: np.ndarray,
    model_name: str,
    gt_csv_path: Optional[Union[str, Path]],
    window: int = 0,
) -> BaselineResult:
    """Attach integer codes, optional public-window evaluation metrics, and full-match diagnostics."""
    raw_codes = np.where(labels == "H", HOME, np.where(labels == "A", AWAY, NOBODY)).astype(np.int8)

    metrics = None
    gt_path = _resolve_gt_path(gt_csv_path)
    if gt_path is not None:
        gt_codes = pm.label_codes(pm.read_label_csv(str(gt_path)))
        pred_codes = pm.label_codes(labels)
        metrics = pm.evaluate(gt_codes, pred_codes)

    diagnostics = diagnose_prediction(labels, model_name=model_name)
    return BaselineResult(
        labels=labels,
        raw_codes=raw_codes,
        metrics=metrics,
        diagnostics=diagnostics,
        window=window,
    )


def _base_deadball_candidate(kin: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute decoupled dead-ball candidate mask from tempo slowdown, OOB hold, and ball state."""
    slowdown_diff = kin["slowdown_diff"]
    bp_s = kin["bp_s"]
    return (
        ((slowdown_diff < -0.78) & kin["ball_not_fast"])
        | ((bp_s < 0.25) & (slowdown_diff < -0.62))
        | kin["oob_causal"]
        | kin["uncontrolled_slow"]
    )


def _compute_ablation2_scores(
    data: MatchData,
    kin: Dict[str, np.ndarray],
    tag_df: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous H/A score, dead-ball score, and binary D mask for Ablations 2 & 3."""
    tag_prior, sp_act = compute_smart_tagging_priors(len(data.frames), tag_df)

    valid_ball = kin["valid_ball"]
    d_H = kin["d_H"]
    d_A = kin["d_A"]
    d_min = kin["d_min"]
    bs_clean = kin["bs_clean"]

    raw_margin = np.zeros(len(data.frames), dtype=np.float64)
    raw_margin[valid_ball] = np.tanh((d_A[valid_ball] - d_H[valid_ball]) / 1.5)

    # Weight H/A margins by ball-control confidence so mid-flight passes do not flip possession
    w_dist = np.exp(-np.maximum(0.0, d_min - 1.5) / 2.5)
    w_speed = np.exp(-np.maximum(0.0, bs_clean - 8.0) / 8.0)
    w_ctrl = np.where(valid_ball, np.clip(w_dist * w_speed, 0.05, 1.0), 0.0)

    num_nw = box_smooth_half(w_ctrl * raw_margin, 38)
    den_nw = box_smooth_half(w_ctrl, 38)
    nw_margin = np.where(den_nw > 1e-4, num_nw / np.maximum(den_nw, 1e-4), 0.0)
    nw_margin = _forward_backfill_mask(nw_margin, den_nw > 1e-4)

    w_tag = np.where(den_nw < 0.10, 0.35, np.where(den_nw < 0.25, 0.15, 0.05))
    ha_score2 = nw_margin + w_tag * tag_prior

    slowdown_diff = kin["slowdown_diff"]
    bp_s = kin["bp_s"]
    bs_s = kin["bs_s"]
    oob_causal = kin["oob_causal"]

    d_cand1 = _base_deadball_candidate(kin)
    d_cand2 = d_cand1 | ((sp_act > 0.0) & (slowdown_diff < -0.62) & (bs_s < 3.2))
    d_mask2 = prune_binary_runs(d_cand2, min_true=115, min_false=65)

    d_score2 = (
        -(slowdown_diff + 0.76)
        + 0.22 * (1.0 - bp_s)
        + 0.14 * sp_act
        + 1.2 * oob_causal.astype(np.float64)
    )
    return ha_score2, d_score2, d_mask2


def run_ablation1_decoupled(
    data: Optional[MatchData] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Ablation 1: Decoupled tempo-normalized D detection + H/A carry-forward & min-dwell."""
    if data is None:
        data = load_match_data()

    kin = compute_kinematic_signals(data)
    d_H = kin["d_H"]
    d_A = kin["d_A"]

    # Stage 1A: Dead-ball detection decoupled from ball dropouts
    d_cand1 = _base_deadball_candidate(kin)
    d_mask1 = prune_binary_runs(d_cand1, min_true=115, min_false=65)

    # Stage 1B: H/A attribution with carry-forward across missing-ball frames & min-dwell
    ha_raw = np.zeros(len(data.frames), dtype=np.float64)
    ha_raw[data.ball_present] = np.where(d_H[data.ball_present] <= d_A[data.ball_present], 1.0, -1.0)
    ha_filled = _forward_backfill_mask(ha_raw, data.ball_present)
    ha_s1 = box_smooth_half(ha_filled, 45) >= 0.0
    ha_clean1 = prune_binary_runs(ha_s1, min_true=38, min_false=38)

    labels = np.full(len(data.frames), "X", dtype="<U1")
    labels[data.graded_mask] = np.where(
        d_mask1[data.graded_mask],
        "D",
        np.where(ha_clean1[data.graded_mask], "H", "A"),
    )
    labels = cleanup_short_spells(labels, min_spell_frames=25)
    return _finalize_result(labels, "ablation1_decoupled", gt_csv_path, window=45)


def run_ablation2_features_tagging(
    data: Optional[MatchData] = None,
    tag_df: Optional[pl.DataFrame] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Ablation 2: Continuous distance margins, ball-jump gate, and sign-corrected Smart Tagging."""
    if data is None:
        data = load_match_data()
    if tag_df is None:
        tag_df = load_smart_tagging()

    kin = compute_kinematic_signals(data)
    ha_score2, _, d_mask2 = _compute_ablation2_scores(data, kin, tag_df)

    ha_clean2 = prune_binary_runs(ha_score2 >= 0.0, min_true=42, min_false=42)

    labels = np.full(len(data.frames), "X", dtype="<U1")
    labels[data.graded_mask] = np.where(
        d_mask2[data.graded_mask],
        "D",
        np.where(ha_clean2[data.graded_mask], "H", "A"),
    )
    labels = cleanup_short_spells(labels, min_spell_frames=25)
    return _finalize_result(labels, "ablation2_features_tags", gt_csv_path, window=38)


def run_ablation3_sequence_decoder(
    data: Optional[MatchData] = None,
    tag_df: Optional[pl.DataFrame] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Ablation 3: Joint 3-state (H, A, D) Viterbi sequence decoder over engineered features."""
    if data is None:
        data = load_match_data()
    if tag_df is None:
        tag_df = load_smart_tagging()

    kin = compute_kinematic_signals(data)
    ha_score2, _, d_mask2 = _compute_ablation2_scores(data, kin, tag_df)

    emit_ha = box_smooth_half(ha_score2, 14)
    emit_d = np.where(d_mask2, 1.30, -0.55)
    emit = np.column_stack([emit_ha, -emit_ha, emit_d])

    trans_mat = np.array(
        [
            [0.0, -16.0, -24.0],
            [-16.0, 0.0, -24.0],
            [-24.0, -24.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels = np.full(len(data.frames), "X", dtype="<U1")
    state_map = np.array(["H", "A", "D"], dtype="<U1")
    for start, stop in HALF_RANGES:
        path = viterbi_decode_half(emit[start:stop], trans_mat)
        labels[start:stop] = state_map[path]

    labels = cleanup_short_spells(labels, min_spell_frames=25)
    return _finalize_result(labels, "ablation3_viterbi", gt_csv_path, window=14)


def run_all_ablations(
    data: Optional[MatchData] = None,
    tag_df: Optional[pl.DataFrame] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
    save_best_to: Optional[Union[str, Path]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, BaselineResult]]:
    """Run baseline benchmarks + all 3 ablations and compare on full match + public window."""
    if data is None:
        data = load_match_data()
    if tag_df is None:
        tag_df = load_smart_tagging()

    results: Dict[str, BaselineResult] = {
        "baseline_raw": run_baseline_pipeline(data=data, window=0, gt_csv_path=gt_csv_path),
        "baseline_smooth25": run_baseline_pipeline(data=data, window=25, gt_csv_path=gt_csv_path),
        "baseline_smooth50": run_baseline_pipeline(data=data, window=50, gt_csv_path=gt_csv_path),
        "baseline_smooth125": run_baseline_pipeline(data=data, window=125, gt_csv_path=gt_csv_path),
        "ablation1_decoupled": run_ablation1_decoupled(data=data, gt_csv_path=gt_csv_path),
        "ablation2_features_tags": run_ablation2_features_tagging(
            data=data, tag_df=tag_df, gt_csv_path=gt_csv_path
        ),
        "ablation3_viterbi": run_ablation3_sequence_decoder(
            data=data, tag_df=tag_df, gt_csv_path=gt_csv_path
        ),
    }

    if save_best_to is not None:
        save_submission(
            results["ablation3_viterbi"].labels,
            save_best_to,
            ungraded_to_x=True,
            graded_mask=data.graded_mask,
        )

    labels_map = {name: res.labels for name, res in results.items()}
    summary_df, detailed_df = compare_models_full_match(labels_map, gt_csv_path=gt_csv_path)
    return summary_df, detailed_df, results


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(240)

    out_sub = Path("data/predictions/ablation3_viterbi.csv")
    summary_df, detailed_df, _ = run_all_ablations(save_best_to=out_sub)

    print("=== 3-STAGE ABLATION STUDY: FULL-MATCH VS PUBLIC 5-MIN WINDOW ===")
    print(summary_df)

    focus_models = [
        "GT (Public 5-min)",
        "ablation1_decoupled",
        "ablation2_features_tags",
        "ablation3_viterbi",
    ]
    print("\n=== PER-SLICE BREAKDOWN (GT + 3 ABLATION STAGES) ===")
    print(detailed_df.filter(pl.col("model").is_in(focus_models)))
    print(f"\nSaved validated submission to: {out_sub}")

