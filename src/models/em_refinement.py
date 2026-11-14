"""Step 3 temporal context, physics-bounded short stoppage detection, and Outer-EM refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from src.data import MatchData, load_match_data
from src.features import PossessionFeatures, build_possession_features
from src.metrics import possession_metrics as pm
from src.models.hsmm_decoder import decode_hsmm_3state
from src.models.weak_emissions import (
    BALL_DROP_MASK_COLS,
    HALF_RANGES,
    _augment_occlusion_copies,
    _build_ha_weak_anchors,
    _fit_lr_hgb_expert,
    compute_carried_margin,
    prob_to_tanh_emission,
    smooth_per_half,
)

STEP3_OCC_COLS: Tuple[str, ...] = (
    "slowdown_diff",
    "bp_s",
    "oob_causal",
    "uncontrolled_slow",
    "all_vx_s",
    "all_vx_rm50",
    "h_vx_s",
    "a_vx_s",
    "h_mean_x_att",
    "a_mean_x_att",
    "pitch_tilt",
    "pitch_tilt_rm75",
    "tag_prior_dedup",
    "sp_act",
    "bx_att",
    "by_filled",
    "carried_nw_25",
    "carried_nw_50",
    "carried_nw_125",
)

STEP3_BALL_DROP_COLS: Tuple[str, ...] = BALL_DROP_MASK_COLS + (
    "nw_weighted_rm25",
    "nw_rel_v_rm25",
    "nw_support_rm25",
)

STEP3_RICH_COLS: Tuple[str, ...] = STEP3_OCC_COLS + STEP3_BALL_DROP_COLS + (
    "den_nw",
    "kick_impulse",
    "nw_weighted_rm75",
)


@dataclass
class Step3Emissions:
    """Outputs from Step 3 temporal-context Outer-EM self-training and stoppage refinement."""

    p_ha_r1_mix: np.ndarray
    p_ha_em_occ: np.ndarray
    p_ha_em_rich: np.ndarray
    p_ha_em_mix: np.ndarray
    emit_d_enh: np.ndarray
    n_r1_anchors: int
    n_em_frames: int
    n_short_stop_cores: int
    anchor_home_share: float


def _erode_per_half(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a boolean mask independently within each half to exclude transitions."""
    out = np.zeros_like(mask, dtype=bool)
    struct = np.ones(2 * radius + 1, dtype=bool)
    for start, stop in HALF_RANGES:
        out[start:stop] = binary_erosion(mask[start:stop], structure=struct, border_value=False)
    return out


def build_step3_feature_matrix(pf: PossessionFeatures) -> Tuple[np.ndarray, List[str]]:
    """Expand base Step 1 features with multi-horizon carried and rolling temporal context."""
    X_base, base_names = pf.to_matrix()
    f = pf.features
    weighted_nw = pf.nw_margin * pf.den_nw

    extra = [
        compute_carried_margin(pf.nw_margin, pf.den_nw, half_life=25),
        compute_carried_margin(pf.nw_margin, pf.den_nw, half_life=50),
        compute_carried_margin(pf.nw_margin, pf.den_nw, half_life=125),
        smooth_per_half(weighted_nw, 25),
        smooth_per_half(weighted_nw, 75),
        smooth_per_half(f["all_vx_s"], 50),
        smooth_per_half(f["pitch_tilt"], 75),
        smooth_per_half(f["nw_rel_v"], 25),
        smooth_per_half(f["nw_support"], 25),
    ]
    extra_names = [
        "carried_nw_25",
        "carried_nw_50",
        "carried_nw_125",
        "nw_weighted_rm25",
        "nw_weighted_rm75",
        "all_vx_rm50",
        "pitch_tilt_rm75",
        "nw_rel_v_rm25",
        "nw_support_rm25",
    ]
    return np.column_stack([X_base] + extra), list(base_names) + extra_names


def build_enhanced_d_emission(
    pf: PossessionFeatures,
    ball_present: np.ndarray,
    short_boost: float = 1.60,
) -> Tuple[np.ndarray, int]:
    """Recover brief ball-absent collective-walking stoppages missed by morphological opening.

    Triggered only when collective speed slowdown (< -0.75 m/s), slow-player fraction (> +0.30),
    and local ball presence (< 0.35) coincide for >= 1.4s (35 frames), then grown solely across
    adjacent missing-ball slow frames so restart frames stay sharp.
    """
    f = pf.features
    base_d = pf.d_mask.astype(bool)
    core = (
        (f["slowdown_diff"] < -0.75)
        & (f["frac_slow_diff"] > 0.30)
        & (f["bp_s"] < 0.35)
        & (~base_d)
    )
    segs = [
        seg
        for seg in pm.segments(core.astype(np.int32), list(HALF_RANGES))
        if seg[2] == 1 and (seg[1] - seg[0]) >= 35
    ]
    short_core = np.zeros_like(base_d)
    for start, stop, _ in segs:
        short_core[start:stop] = True

    not_live_ball = (~ball_present.astype(bool)) & (f["bs_clean"] < 6.0)
    grown = short_core.copy()
    for _ in range(35):
        grown = binary_dilation(grown, iterations=1) & (not_live_ball | short_core)

    emit_d = np.where(base_d, 1.25, -1.25)
    emit_d[grown] = float(short_boost)
    return emit_d, len(segs)


def _fit_step3_round(
    X_full: np.ndarray,
    name_to_idx: Dict[str, int],
    mask: np.ndarray,
    y_home: np.ndarray,
    sample_conf: np.ndarray | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit regularized occlusion + rich dropout experts with optional confidence weights."""
    y_tr = y_home[mask]
    home_share = float(y_tr.mean())
    sw = np.where(
        y_tr == 1,
        0.5 / max(1e-6, home_share),
        0.5 / max(1e-6, 1.0 - home_share),
    )
    if sample_conf is not None:
        sw = sw * sample_conf[mask]

    y_stack = np.concatenate([y_tr, y_tr])
    sw_stack = np.concatenate([sw, 0.8 * sw])
    rng = np.random.default_rng(seed)

    occ_idx = [name_to_idx[c] for c in STEP3_OCC_COLS]
    X_occ_clean = X_full[mask][:, occ_idx]
    X_occ_aug = _augment_occlusion_copies(
        X_occ_clean,
        STEP3_OCC_COLS,
        rng,
        mask_ball_local=False,
        carried_cols=("carried_nw_25", "carried_nw_50", "carried_nw_125"),
    )
    p_occ = _fit_lr_hgb_expert(
        np.vstack([X_occ_clean, X_occ_aug]),
        y_stack,
        sw_stack,
        X_full[:, occ_idx],
        seed=seed,
        c_reg=0.20,
        max_iter=160,
        max_depth=4,
        min_samples_leaf=150,
        l2_regularization=4.0,
    )

    rich_idx = [name_to_idx[c] for c in STEP3_RICH_COLS]
    X_rich_clean = X_full[mask][:, rich_idx]
    X_rich_aug = _augment_occlusion_copies(
        X_rich_clean,
        STEP3_RICH_COLS,
        rng,
        mask_ball_local=True,
        carried_cols=("carried_nw_25", "carried_nw_50"),
        ball_mask_cols=STEP3_BALL_DROP_COLS,
        slow_carry_col="nw_weighted_rm75",
    )
    p_rich = _fit_lr_hgb_expert(
        np.vstack([X_rich_clean, X_rich_aug]),
        y_stack,
        sw_stack,
        X_full[:, rich_idx],
        seed=seed + 1,
        c_reg=0.20,
        max_iter=160,
        max_depth=4,
        min_samples_leaf=150,
        l2_regularization=4.0,
    )
    return p_occ, p_rich


def fit_predict_step3_em(
    data: MatchData | None = None,
    pf: PossessionFeatures | None = None,
    seed: int = 42,
) -> Step3Emissions:
    """Run Round 1 weak-anchor fit, HSMM decode, and Round 2 Outer-EM self-training."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)

    # Strict zero-leakage protocol: never fit Round 1 or Round 2 on the 5-min public GT window.
    train_mask = data.graded_mask & (~data.win_mask)
    anchor_r1, y_r1 = _build_ha_weak_anchors(pf, train_mask=train_mask)

    X_s3, s3_names = build_step3_feature_matrix(pf)
    name_to_idx = {name: i for i, name in enumerate(s3_names)}
    emit_d_enh, n_short_cores = build_enhanced_d_emission(pf, data.ball_present, short_boost=1.60)

    # Round 1 fit -> HSMM sequence scaffold
    p_occ_r1, p_rich_r1 = _fit_step3_round(X_s3, name_to_idx, anchor_r1, y_r1, sample_conf=None, seed=seed)
    p_mix_r1 = 0.5 * p_occ_r1 + 0.5 * p_rich_r1

    lbl_r1_hsmm = decode_hsmm_3state(
        n_frames=len(data.frames),
        emit_ha=prob_to_tanh_emission(p_mix_r1, temperature=2.5, smooth_width=11),
        emit_d=emit_d_enh,
        pen_ha=-24.0,
        pen_d=-26.0,
        min_dur_ha_frames=25,
        min_dur_d_frames=45,
        dur_weight=0.7,
    )

    # Outer-EM expansion across confident non-window H/A sequence interiors
    is_h = _erode_per_half(lbl_r1_hsmm == "H", radius=15)
    is_a = _erode_per_half(lbl_r1_hsmm == "A", radius=15)
    seq_interior = (is_h | is_a) & train_mask
    em_conf = np.where(is_h, p_mix_r1, 1.0 - p_mix_r1)
    em_mask = anchor_r1 | (seq_interior & (em_conf >= 0.72))
    y_em = np.where(anchor_r1, y_r1, is_h.astype(np.int32))
    sample_conf = np.where(anchor_r1, 1.0, 0.65)

    p_occ_em, p_rich_em = _fit_step3_round(
        X_s3,
        name_to_idx,
        em_mask,
        y_em,
        sample_conf=sample_conf,
        seed=seed + 59,
    )
    p_mix_em = 0.5 * p_occ_em + 0.5 * p_rich_em

    return Step3Emissions(
        p_ha_r1_mix=p_mix_r1,
        p_ha_em_occ=p_occ_em,
        p_ha_em_rich=p_rich_em,
        p_ha_em_mix=p_mix_em,
        emit_d_enh=emit_d_enh,
        n_r1_anchors=int(anchor_r1.sum()),
        n_em_frames=int(em_mask.sum()),
        n_short_stop_cores=n_short_cores,
        anchor_home_share=float(y_r1[anchor_r1].mean()),
    )

