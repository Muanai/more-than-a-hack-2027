"""Zero-leakage weakly-supervised emission models for Step 2 possession segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import MatchData, load_match_data
from src.data.loader import HALF1_END, HALF1_START, HALF2_END, HALF2_START
from src.features import PossessionFeatures, build_possession_features

HALF_RANGES: Tuple[Tuple[int, int], ...] = (
    (HALF1_START, HALF1_END + 1),
    (HALF2_START, HALF2_END + 1),
)

OCCLUSION_EXPERT_COLS: Tuple[str, ...] = (
    "slowdown_diff",
    "bp_s",
    "oob_causal",
    "uncontrolled_slow",
    "all_vx_s",
    "h_vx_s",
    "a_vx_s",
    "h_mean_x_att",
    "a_mean_x_att",
    "pitch_tilt",
    "tag_prior_dedup",
    "sp_act",
    "bx_att",
    "by_filled",
    "carried_nw",
)

BALL_DROP_MASK_COLS: Tuple[str, ...] = (
    "nw_margin",
    "nw_rel_v",
    "nw_support",
    "rel_v_margin",
    "d_margin2",
)

RICH_EXPERT_COLS: Tuple[str, ...] = OCCLUSION_EXPERT_COLS + BALL_DROP_MASK_COLS + (
    "den_nw",
    "kick_impulse",
)


@dataclass
class LearnedEmissions:
    """Outputs from zero-leakage weakly-supervised H-vs-A and stoppage emitters."""

    p_ha_occ: np.ndarray
    p_ha_rich: np.ndarray
    p_ha_mix: np.ndarray
    emit_d_morph: np.ndarray
    n_ha_anchors: int
    ha_anchor_home_share: float


def compute_carried_margin(
    nw_margin: np.ndarray,
    den_nw: np.ndarray,
    half_life: int = 50,
    min_density: float = 0.35,
) -> np.ndarray:
    """Propagate confident NW control margin forward across occlusion gaps with decay."""
    out = np.zeros(len(nw_margin), dtype=np.float64)
    decay = np.exp(-np.log(2.0) / float(half_life))
    for start, stop in HALF_RANGES:
        val = 0.0
        for i in range(start, stop):
            if den_nw[i] >= min_density:
                val = 0.25 * val + 0.75 * float(nw_margin[i])
            else:
                val *= decay
            out[i] = val
    return out


def smooth_per_half(values: np.ndarray, width: int) -> np.ndarray:
    """Apply centered moving-average smoothing strictly within each match half."""
    if width <= 1:
        return values.astype(np.float64, copy=True)
    out = np.zeros_like(values, dtype=np.float64)
    for start, stop in HALF_RANGES:
        out[start:stop] = uniform_filter1d(
            values[start:stop].astype(np.float64),
            size=width,
            mode="nearest",
        )
    return out


def prob_to_tanh_emission(
    prob: np.ndarray,
    temperature: float = 2.5,
    smooth_width: int = 1,
    eps: float = 1e-4,
) -> np.ndarray:
    """Map probabilities to bounded log-odds emissions compatible with Viterbi/HSMM scales."""
    p = np.clip(prob, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    squashed = np.tanh(logit / float(temperature))
    return smooth_per_half(squashed, width=smooth_width)


def _fit_lr_hgb_expert(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_eval: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Fit a balanced LogisticRegression + HistGradientBoosting ensemble."""
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=0.25, max_iter=500, random_state=seed)
    lr.fit(X_tr_std, y_train, sample_weight=sample_weight)

    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=80,
        random_state=seed,
    )
    hgb.fit(X_train, y_train, sample_weight=sample_weight)

    p_lr = lr.predict_proba(scaler.transform(X_eval))[:, 1]
    p_hgb = hgb.predict_proba(X_eval)[:, 1]
    return 0.4 * p_lr + 0.6 * p_hgb


def _build_ha_weak_anchors(
    pf: PossessionFeatures,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select high-precision H-vs-A weak-supervision frames outside the public window."""
    d_mask = pf.d_mask.astype(bool)
    nw = pf.nw_margin
    den = pf.den_nw
    tag = pf.tag_prior_dedup

    strong_nw = (~d_mask) & (den >= 0.35) & (np.abs(nw) >= 0.32) & ((nw * tag) >= -0.05)
    strong_tag = (~d_mask) & (np.abs(tag) >= 0.75) & ((nw * tag) >= -0.25)
    anchor_mask = train_mask & (strong_nw | strong_tag)

    combined_signal = np.where(np.abs(nw) * den >= 0.15, nw, tag)
    y_home = (combined_signal > 0).astype(np.int32)
    return anchor_mask, y_home


def _augment_occlusion_copies(
    X_clean: np.ndarray,
    col_names: Sequence[str],
    rng: np.random.Generator,
    mask_ball_local: bool,
) -> np.ndarray:
    """Synthesize occluded training copies by decaying carried margin and masking local cues."""
    X_aug = X_clean.copy()
    name_to_idx = {name: i for i, name in enumerate(col_names)}

    if mask_ball_local:
        drop_ball = rng.random(len(X_aug)) < 0.50
        for col in BALL_DROP_MASK_COLS:
            X_aug[drop_ball, name_to_idx[col]] = 0.0
        X_aug[drop_ball, name_to_idx["den_nw"]] = 0.0
        c_idx = name_to_idx["carried_nw"]
        X_aug[drop_ball, c_idx] *= rng.uniform(0.1, 0.6, size=int(drop_ball.sum()))
    else:
        drop_cnw = rng.random(len(X_aug)) < 0.45
        c_idx = name_to_idx["carried_nw"]
        X_aug[drop_cnw, c_idx] *= rng.uniform(0.0, 0.3, size=int(drop_cnw.sum()))

    drop_tag = rng.random(len(X_aug)) < 0.35
    X_aug[drop_tag, name_to_idx["tag_prior_dedup"]] = 0.0
    return X_aug


def fit_predict_weak_emissions(
    data: MatchData | None = None,
    pf: PossessionFeatures | None = None,
    seed: int = 42,
) -> LearnedEmissions:
    """Fit zero-leakage H-vs-A dropout experts strictly on graded non-window frames."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)

    # Strict zero-leakage protocol: never fit on the 5-min public GT window.
    train_mask = data.graded_mask & (~data.win_mask)
    anchor_mask, y_home = _build_ha_weak_anchors(pf, train_mask=train_mask)

    X_base, base_names = pf.to_matrix()
    carried_nw = compute_carried_margin(pf.nw_margin, pf.den_nw, half_life=50)
    X_full = np.column_stack([X_base, carried_nw])
    full_names = list(base_names) + ["carried_nw"]
    name_to_idx = {name: i for i, name in enumerate(full_names)}

    y_anc = y_home[anchor_mask]
    home_share = float(y_anc.mean())
    pos_w = 0.5 / max(1e-6, home_share)
    neg_w = 0.5 / max(1e-6, 1.0 - home_share)
    sw = np.where(y_anc == 1, pos_w, neg_w)

    y_stack = np.concatenate([y_anc, y_anc])
    sw_stack = np.concatenate([sw, 0.8 * sw])
    rng = np.random.default_rng(seed)

    occ_idx = [name_to_idx[c] for c in OCCLUSION_EXPERT_COLS]
    X_occ_clean = X_full[anchor_mask][:, occ_idx]
    X_occ_aug = _augment_occlusion_copies(X_occ_clean, OCCLUSION_EXPERT_COLS, rng, mask_ball_local=False)
    p_occ = _fit_lr_hgb_expert(
        np.vstack([X_occ_clean, X_occ_aug]),
        y_stack,
        sw_stack,
        X_full[:, occ_idx],
        seed=seed,
    )

    rich_idx = [name_to_idx[c] for c in RICH_EXPERT_COLS]
    X_rich_clean = X_full[anchor_mask][:, rich_idx]
    X_rich_aug = _augment_occlusion_copies(X_rich_clean, RICH_EXPERT_COLS, rng, mask_ball_local=True)
    p_rich = _fit_lr_hgb_expert(
        np.vstack([X_rich_clean, X_rich_aug]),
        y_stack,
        sw_stack,
        X_full[:, rich_idx],
        seed=seed + 1,
    )

    emit_d_morph = np.where(pf.d_mask.astype(bool), 1.25, -1.25)
    return LearnedEmissions(
        p_ha_occ=p_occ,
        p_ha_rich=p_rich,
        p_ha_mix=0.5 * p_occ + 0.5 * p_rich,
        emit_d_morph=emit_d_morph,
        n_ha_anchors=int(anchor_mask.sum()),
        ha_anchor_home_share=home_share,
    )

