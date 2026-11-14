"""Calibrated 3-state Viterbi and explicit-duration HSMM sequence decoders."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.data.loader import HALF1_END, HALF1_START, HALF2_END, HALF2_START
from src.pipelines.ablation import cleanup_short_spells, viterbi_decode_half

HALF_RANGES: Tuple[Tuple[int, int], ...] = (
    (HALF1_START, HALF1_END + 1),
    (HALF2_START, HALF2_END + 1),
)
STATE_MAP = np.array(["H", "A", "D"], dtype="<U1")


def _build_trans_mat(pen_ha: float, pen_d: float) -> np.ndarray:
    return np.array(
        [
            [0.0, pen_ha, pen_d],
            [pen_ha, 0.0, pen_d],
            [pen_d, pen_d, 0.0],
        ],
        dtype=np.float64,
    )


def decode_viterbi_3state(
    n_frames: int,
    emit_ha: np.ndarray,
    emit_d: np.ndarray,
    pen_ha: float = -22.0,
    pen_d: float = -24.0,
    min_spell_frames: int = 25,
) -> np.ndarray:
    """Decode H/A/D sequence independently per half using 3-state Viterbi DP."""
    emit = np.column_stack([emit_ha, -emit_ha, emit_d])
    trans_mat = _build_trans_mat(pen_ha, pen_d)
    labels = np.full(n_frames, "X", dtype="<U1")
    for start, stop in HALF_RANGES:
        path = viterbi_decode_half(emit[start:stop], trans_mat)
        labels[start:stop] = STATE_MAP[path]
    return cleanup_short_spells(labels, min_spell_frames=min_spell_frames)


def _compute_duration_prior(
    max_blocks: int,
    block_frames: int,
    dur_weight: float,
    mean_ha_s: float = 9.0,
    mean_d_s: float = 12.0,
) -> np.ndarray:
    """Build weak log-normal dwell-time shaping priors for states H, A, D."""
    dur_prior = np.zeros((3, max_blocks + 1), dtype=np.float64)
    if dur_weight <= 0.0:
        return dur_prior

    d_sec = np.arange(1, max_blocks + 1, dtype=np.float64) * (block_frames / 25.0)
    lp_ha = -0.5 * ((np.log(np.maximum(d_sec, 0.2)) - np.log(mean_ha_s)) / 1.15) ** 2
    lp_d = -0.5 * ((np.log(np.maximum(d_sec, 0.2)) - np.log(mean_d_s)) / 1.15) ** 2
    dur_prior[0, 1:] = dur_weight * (lp_ha - lp_ha.max())
    dur_prior[1, 1:] = dur_weight * (lp_ha - lp_ha.max())
    dur_prior[2, 1:] = dur_weight * (lp_d - lp_d.max())
    return dur_prior


def _hsmm_decode_segment(
    seg_emit: np.ndarray,
    trans_mat: np.ndarray,
    min_blocks: np.ndarray,
    max_blocks: int,
    dur_prior: np.ndarray,
    block_frames: int,
) -> np.ndarray:
    """Run exact explicit-duration Viterbi DP over 0.2s blocks for one half."""
    n_seg = len(seg_emit)
    n_blk = (n_seg + block_frames - 1) // block_frames
    pad = n_blk * block_frames - n_seg
    seg_pad = (
        np.vstack([seg_emit, np.repeat(seg_emit[-1:], pad, axis=0)])
        if pad > 0
        else seg_emit
    )
    blk_emit = seg_pad.reshape(n_blk, block_frames, 3).sum(axis=1)

    pref = np.zeros((n_blk + 1, 3), dtype=np.float64)
    pref[1:] = np.cumsum(blk_emit, axis=0)

    dp = np.full((n_blk + 1, 3), -1e18, dtype=np.float64)
    back_start = np.zeros((n_blk + 1, 3), dtype=np.int32)
    dp[0, :] = 0.0

    enter_val = np.full((n_blk + 1, 3), -1e18, dtype=np.float64)
    enter_val[0, :] = 0.0

    for t in range(1, n_blk + 1):
        for st in range(3):
            d_lo = int(min_blocks[st])
            if t < d_lo:
                continue
            u_lo = max(0, t - max_blocks)
            u_idx = np.arange(u_lo, t - d_lo + 1)
            durs = t - u_idx
            scores = enter_val[u_idx, st] + (pref[t, st] - pref[u_idx, st]) + dur_prior[st, durs]
            best_k = int(np.argmax(scores))
            dp[t, st] = scores[best_k]
            back_start[t, st] = u_idx[best_k]

        if t < n_blk:
            for nxt in range(3):
                best_v = -1e18
                for prv in range(3):
                    if prv != nxt:
                        val = dp[t, prv] + trans_mat[prv, nxt]
                        if val > best_v:
                            best_v = val
                enter_val[t, nxt] = best_v

    path_blk = np.empty(n_blk, dtype=np.int32)
    t = n_blk
    st = int(np.argmax(dp[n_blk]))
    while t > 0:
        u = int(back_start[t, st])
        path_blk[u:t] = st
        if u == 0:
            break
        best_v = -1e18
        best_prev = 0
        for prv in range(3):
            if prv != st:
                val = dp[u, prv] + trans_mat[prv, st]
                if val > best_v:
                    best_v = val
                    best_prev = prv
        t = u
        st = best_prev

    return np.repeat(path_blk, block_frames)[:n_seg]


def decode_hsmm_3state(
    n_frames: int,
    emit_ha: np.ndarray,
    emit_d: np.ndarray,
    pen_ha: float = -22.0,
    pen_d: float = -26.0,
    min_dur_ha_frames: int = 25,
    min_dur_d_frames: int = 45,
    max_dur_frames: int = 1200,
    dur_weight: float = 0.7,
    block_frames: int = 5,
    min_spell_frames: int = 25,
) -> np.ndarray:
    """Decode H/A/D sequence per half via explicit-duration HSMM over 0.2s blocks."""
    emit = np.column_stack([emit_ha, -emit_ha, emit_d])
    trans_mat = _build_trans_mat(pen_ha, pen_d)
    min_blocks = np.array([
        max(1, min_dur_ha_frames // block_frames),
        max(1, min_dur_ha_frames // block_frames),
        max(1, min_dur_d_frames // block_frames),
    ])
    max_blocks = max_dur_frames // block_frames
    dur_prior = _compute_duration_prior(max_blocks, block_frames, dur_weight=dur_weight)

    labels = np.full(n_frames, "X", dtype="<U1")
    for start, stop in HALF_RANGES:
        path = _hsmm_decode_segment(
            emit[start:stop],
            trans_mat=trans_mat,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            dur_prior=dur_prior,
            block_frames=block_frames,
        )
        labels[start:stop] = STATE_MAP[path]
    return cleanup_short_spells(labels, min_spell_frames=min_spell_frames)

