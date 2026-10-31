import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.metrics import possession_metrics as pm

    PITCH_LENGTH = 105.3
    PITCH_WIDTH = 68.0
    FPS = 25.0

    # Graded match blocks (inclusive video frames)
    HALF1_START, HALF1_END = 11185, 90651
    HALF2_START, HALF2_END = 110808, 190197
    WIN_START, WIN_END = 11185, 18684
    N_FRAMES = 199319
    return (
        FPS,
        HALF1_END,
        HALF1_START,
        HALF2_END,
        HALF2_START,
        N_FRAMES,
        PITCH_LENGTH,
        PITCH_WIDTH,
        ROOT,
        WIN_END,
        WIN_START,
        json,
        mo,
        np,
        pl,
        pm,
    )


@app.cell
def _(
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    N_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    ROOT,
    WIN_END,
    WIN_START,
    json,
    mo,
    np,
    pl,
    pm,
):
    # =========================================================================
    # §1. LOAD + REGRESSION ASSERTIONS
    # =========================================================================
    tracking_df = pl.read_parquet(ROOT / "data" / "processed" / "tactical" / "tracking.parquet")
    slots_df = pl.read_parquet(ROOT / "data" / "processed" / "tactical" / "player_slots.parquet")
    tagging_df = pl.read_parquet(ROOT / "data" / "processed" / "tactical" / "smart_tagging.parquet")
    homography_df = pl.read_parquet(ROOT / "data" / "processed" / "tactical" / "homography.parquet")

    gt_csv_path = ROOT / "reference" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt_meta_path = ROOT / "reference" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public_meta.json"
    with open(gt_meta_path, "r", encoding="utf-8") as _f:
        gt_meta = json.load(_f)

    gt_raw_df = pm.read_label_csv(str(gt_csv_path))
    gt_codes = pm.label_codes(gt_raw_df)
    gt_runs = pm.valid_runs(gt_codes)
    gt_segs = pm.segments(gt_codes, gt_runs)
    gt_spells = pm.possession_spells(gt_codes, gt_runs)

    # Precompute core NumPy matrices for fast vectorized geometry across all cells
    frames_np = tracking_df["frame"].to_numpy()
    ball_x_norm = tracking_df["ball_x"].to_numpy()
    ball_y_norm = tracking_df["ball_y"].to_numpy()
    ball_speed_raw = tracking_df["ball_speed"].to_numpy()
    ball_present = ~np.isnan(ball_x_norm) & ~np.isnan(ball_y_norm)

    home_slots = slots_df.filter(pl.col("team") == "H")["slot_id"].to_list()
    away_slots = slots_df.filter(pl.col("team") == "A")["slot_id"].to_list()

    px_norm = np.column_stack([tracking_df[f"p{i}_x"].to_numpy() for i in range(1, 47)])
    py_norm = np.column_stack([tracking_df[f"p{i}_y"].to_numpy() for i in range(1, 47)])
    ps_raw = np.column_stack([tracking_df[f"p{i}_speed"].to_numpy() for i in range(1, 47)])

    slot_is_home = np.array([1 if (i + 1) in home_slots else 0 for i in range(46)], dtype=bool)
    slot_is_away = ~slot_is_home

    active_players_per_frame = np.sum(~np.isnan(px_norm), axis=1)
    active_home_per_frame = np.sum(~np.isnan(px_norm[:, slot_is_home]), axis=1)
    active_away_per_frame = np.sum(~np.isnan(px_norm[:, slot_is_away]), axis=1)

    # Phase masks
    graded_mask = np.zeros(N_FRAMES, dtype=bool)
    graded_mask[HALF1_START : HALF1_END + 1] = True
    graded_mask[HALF2_START : HALF2_END + 1] = True

    win_mask = np.zeros(N_FRAMES, dtype=bool)
    win_mask[WIN_START : WIN_END + 1] = True

    # Nearest distances in metres (105.3 x 68.0)
    dx_m = (px_norm - ball_x_norm[:, None]) * PITCH_LENGTH
    dy_m = (py_norm - ball_y_norm[:, None]) * PITCH_WIDTH
    dist_m = np.hypot(dx_m, dy_m)

    dist_home_m = np.where(slot_is_home[None, :], dist_m, np.inf)
    dist_away_m = np.where(slot_is_away[None, :], dist_m, np.inf)
    with np.errstate(all="ignore"):
        d_H = np.nanmin(np.where( np.isnan(dist_home_m), np.inf, dist_home_m ), axis=1)
        d_A = np.nanmin(np.where( np.isnan(dist_away_m), np.inf, dist_away_m ), axis=1)
    d_H[~ball_present] = np.nan
    d_A[~ball_present] = np.nan

    # Run all 13 regression assertions from §1
    _n_graded = int(graded_mask.sum())
    _n_win = int(win_mask.sum())
    _win_codes = gt_codes[win_mask]
    _h_count = int(np.sum(_win_codes == pm.H))
    _a_count = int(np.sum(_win_codes == pm.A))
    _d_count = int(np.sum(_win_codes == pm.D))
    _mean_poss_s = float(np.mean([b - a for a, b, _ in gt_spells]) / 25.0)
    _ball_avail_all = int(ball_present.sum())
    _ball_avail_graded = int(ball_present[graded_mask].sum())
    _mean_tracked_graded = float(active_players_per_frame[graded_mask].mean())

    assert len(tracking_df) == N_FRAMES, f"Expected {N_FRAMES} tracking rows"
    assert len(homography_df) == N_FRAMES, f"Expected {N_FRAMES} homography rows"
    assert len(gt_raw_df) == N_FRAMES, f"Expected {N_FRAMES} GT rows"
    assert np.array_equal(frames_np, np.arange(N_FRAMES)), "Tracking frame index misaligned"
    assert np.array_equal(homography_df["frame"].to_numpy(), np.arange(N_FRAMES)), "Homography frame index misaligned"
    assert _n_graded == 158857, f"Expected 158857 graded frames, got {_n_graded}"
    assert _n_win == 7500 and gt_runs == [(11185, 18685)], f"Unexpected GT window {gt_runs}"
    assert (_h_count, _a_count, _d_count) == (3454, 2577, 1469), "GT label count mismatch"
    assert (len(gt_segs), len(gt_spells)) == (21, 17), "GT segment/spell count mismatch"
    assert round(_mean_poss_s, 2) == 14.19, f"Expected 14.19s mean spell, got {_mean_poss_s:.3f}"
    assert _ball_avail_all == 113488, f"Expected 113488 ball-present frames, got {_ball_avail_all}"
    assert _ball_avail_graded == 94214, f"Expected 94214 graded ball-present frames, got {_ball_avail_graded}"
    assert (len(slots_df), len(home_slots), len(away_slots)) == (46, 24, 22), "Slot count mismatch"
    assert round(_mean_tracked_graded, 1) == 20.5, f"Expected 20.5 mean tracked objects, got {_mean_tracked_graded:.2f}"
    assert len(tagging_df) == 638, f"Expected 638 smart tagging rows, got {len(tagging_df)}"

    assertions_table = pl.DataFrame([
        {"Check": "Rows in tracking / homography / GT", "Expected": "199,319", "Measured": f"{len(tracking_df):,}", "Status": "PASS"},
        {"Check": "Frame index (0..199,318)", "Expected": "0..199,318", "Measured": f"{frames_np[0]}..{frames_np[-1]}", "Status": "PASS"},
        {"Check": "Graded blocks (11,185-90,651 & 110,808-190,197)", "Expected": "158,857", "Measured": f"{_n_graded:,}", "Status": "PASS"},
        {"Check": "Public GT graded window (11,185-18,684)", "Expected": "7,500", "Measured": f"{_n_win:,}", "Status": "PASS"},
        {"Check": "GT label counts in window (H / A / D)", "Expected": "3,454 / 2,577 / 1,469", "Measured": f"{_h_count:,} / {_a_count:,} / {_d_count:,}", "Status": "PASS"},
        {"Check": "GT segments / possessions in window", "Expected": "21 / 17", "Measured": f"{len(gt_segs)} / {len(gt_spells)}", "Status": "PASS"},
        {"Check": "Mean possession duration", "Expected": "14.19 s", "Measured": f"{_mean_poss_s:.2f} s", "Status": "PASS"},
        {"Check": "Ball available, whole match", "Expected": "113,488 (56.9%)", "Measured": f"{_ball_avail_all:,} ({_ball_avail_all/N_FRAMES*100:.1f}%)", "Status": "PASS"},
        {"Check": "Ball available, graded blocks", "Expected": "94,214 (59.3%)", "Measured": f"{_ball_avail_graded:,} ({_ball_avail_graded/_n_graded*100:.1f}%)", "Status": "PASS"},
        {"Check": "Track slots (Total = Home + Away)", "Expected": "46 = 24 + 22", "Measured": f"{len(slots_df)} = {len(home_slots)} + {len(away_slots)}", "Status": "PASS"},
        {"Check": "Mean tracked objects, graded frames", "Expected": "20.5", "Measured": f"{_mean_tracked_graded:.2f}", "Status": "PASS"},
        {"Check": "Smart Tagging instances", "Expected": "638", "Measured": f"{len(tagging_df)}", "Status": "PASS"},
    ])

    mo.vstack([
        mo.md("# More than a Hack 2027 — Decision-Driven EDA\n## §1. Load & Regression Assertions `[FULL]`\nAll 12 regression checks passed against `DATA.md` and `meta.json`."),
        assertions_table,
    ])
    return (
        active_away_per_frame,
        active_home_per_frame,
        active_players_per_frame,
        ball_present,
        ball_speed_raw,
        ball_x_norm,
        ball_y_norm,
        d_A,
        d_H,
        dist_m,
        graded_mask,
        gt_codes,
        gt_runs,
        gt_segs,
        gt_spells,
        homography_df,
        ps_raw,
        px_norm,
        py_norm,
        slot_is_away,
        slot_is_home,
        slots_df,
        tagging_df,
        win_mask,
    )


@app.cell
def _(
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    active_players_per_frame,
    ball_present,
    ball_x_norm,
    ball_y_norm,
    graded_mask,
    mo,
    np,
    pl,
    win_mask,
):
    # =========================================================================
    # §0.2 Q0.1 — IS THE PUBLIC WINDOW A FAIR SAMPLE OF THE MATCH? [FULL]
    # =========================================================================
    _evidence = "FULL"
    _h1_mask = np.zeros_like(graded_mask)
    _h1_mask[HALF1_START : HALF1_END + 1] = True
    _h2_mask = np.zeros_like(graded_mask)
    _h2_mask[HALF2_START : HALF2_END + 1] = True

    _oob = ball_present & ((ball_x_norm < 0.0) | (ball_x_norm > 1.0) | (ball_y_norm < 0.0) | (ball_y_norm > 1.0))

    def _summarize_slice(name, mask):
        _n = int(mask.sum())
        _bp = int(ball_present[mask].sum())
        _act = active_players_per_frame[mask]
        _oob_bp = int(_oob[mask].sum())
        return {
            "Slice": name,
            "Frames": _n,
            "Ball Avail (%)": round(_bp / _n * 100, 2),
            "Mean Tracked Players": round(float(_act.mean()), 2),
            "Median Tracked": int(np.median(_act)),
            "Frames <10 Players (%)": round(float(np.mean(_act < 10) * 100), 2),
            "Frames >22 Players (%)": round(float(np.mean(_act > 22) * 100), 2),
            "Ball OOB (% of present)": round(_oob_bp / _bp * 100, 2) if _bp else 0.0,
            "Ball OOB (% of all frames)": round(_oob_bp / _n * 100, 2),
        }

    q01_df = pl.DataFrame([
        _summarize_slice("Public 5-min Window (11,185–18,684)", win_mask),
        _summarize_slice("Graded Match Total (1st + 2nd Half)", graded_mask),
        _summarize_slice("1st Half Graded (11,185–90,651)", _h1_mask),
        _summarize_slice("2nd Half Graded (110,808–190,197)", _h2_mask),
        _summarize_slice("Full Video (0–199,318)", np.ones_like(graded_mask, dtype=bool)),
    ])

    mo.vstack([
        mo.md(f"## §0.2 Q0.1 — Is the public window a fair sample of the match? `[{_evidence}]`\n"
              "**Finding:** Ball availability is **88.20%** in the 5-min window vs **59.31%** across graded play (a +48.7% relative discrepancy). "
              "Tracked-player density is slightly higher in the window (**20.99** vs **20.47**, <10 players on **0.00%** of window frames vs **0.24%** of graded play), "
              "and ball out-of-bounds rate among present frames is **3.30%** in the window vs **2.65%** in graded play."),
        q01_df,
    ])
    return


@app.cell
def _(
    FPS,
    ROOT,
    WIN_END,
    WIN_START,
    gt_codes,
    gt_segs,
    gt_spells,
    mo,
    np,
    pl,
    pm,
):
    # =========================================================================
    # §0.3 Q0.2 — WHAT DOES spellF1 ACTUALLY REWARD? [DERIVED]
    # & §0.4 Q0.3 — WHICH SUB-PROBLEM BINDS: ALIVE/DEAD OR HOME/AWAY? [GT-5MIN]
    # =========================================================================
    _rows_q02 = []

    def _eval_perturbed(family, setting, pred_arr):
        _res = pm.evaluate(gt_codes, pred_arr)
        _rows_q02.append({
            "Distortion Family": family,
            "Setting": setting,
            "spellF1 (mean)": round(_res["spell_f1_mean"], 4),
            "spellF1 @ 0.5": round(_res["spell_f1_at_headline"], 4),
            "macroF1": round(_res["macro_f1"], 4),
            "alive_dead_f1": round(_res["alive_dead_f1"], 4),
            "team_acc_both_alive": round(_res["team_acc_both_alive"], 4),
            "n_pred_spells": _res["n_pred_spells"],
            "fragmentation": round(_res["fragmentation"], 2),
        })

    # 1. Uniform boundary shift delta in {0, 1, 3, 8, 25, 50}
    for _delta in (0, 1, 3, 8, 25, 50):
        _p = gt_codes.copy()
        _win = gt_codes[WIN_START : WIN_END + 1]
        if _delta > 0:
            _shifted = np.concatenate([np.full(_delta, _win[0], dtype=np.int8), _win[:-_delta]])
            _p[WIN_START : WIN_END + 1] = _shifted
        _eval_perturbed("1. Boundary shift (δ frames)", f"δ = {_delta}f ({_delta/FPS:.2f}s)", _p)

    # 2. Split every spell into k equal pieces (k = 2, 3, 5, 10) by inserting 1-frame D separator
    for _k in (2, 3, 5, 10):
        _p = gt_codes.copy()
        for _a, _b, _code in gt_spells:
            _len = _b - _a
            if _len >= _k * 2:
                for _cut_idx in range(1, _k):
                    _cut_frame = _a + (_len * _cut_idx) // _k
                    _p[_cut_frame] = pm.D
        _eval_perturbed("2. Spell fragmentation (k pieces)", f"k = {_k}", _p)

    # 3. Merge adjacent same-team spells across an intervening dead-ball run
    _p_merge_same = gt_codes.copy()
    for _idx in range(1, len(gt_segs) - 1):
        _s_start, _s_stop, _s_code = gt_segs[_idx]
        if _s_code == pm.D:
            _prev_code = gt_segs[_idx - 1][2]
            _next_code = gt_segs[_idx + 1][2]
            if _prev_code == _next_code and _prev_code in (pm.H, pm.A):
                _p_merge_same[_s_start:_s_stop] = _prev_code
    _eval_perturbed("3. Failed D call (merge spells)", "Merge same-team across D (2 D-runs)", _p_merge_same)

    _p_kill_all_d = gt_codes.copy()
    for _idx in range(len(gt_segs)):
        _s_start, _s_stop, _s_code = gt_segs[_idx]
        if _s_code == pm.D and _idx > 0:
            _p_kill_all_d[_s_start:_s_stop] = gt_segs[_idx - 1][2]
    _eval_perturbed("3. Failed D call (merge spells)", "Erase all 4 D-runs (carry prev team)", _p_kill_all_d)

    # 4. Drop spells shorter than L in {0.5, 1, 2, 4} s (replace with D)
    for _L_s in (0.5, 1.0, 2.0, 4.0):
        _p = gt_codes.copy()
        _min_f = int(round(_L_s * FPS))
        for _a, _b, _code in gt_spells:
            if (_b - _a) < _min_f:
                _p[_a:_b] = pm.D
        _eval_perturbed("4. Drop short spells (< L sec → D)", f"L = {_L_s:.1f}s ({_min_f}f)", _p)

    # 5. Flip team label on a fraction f in {0.05, 0.1, 0.2} of spells
    _rng = np.random.RandomState(42)
    _perm = _rng.permutation(len(gt_spells))
    for _f in (0.05, 0.10, 0.20):
        _n_flip = max(1, int(round(_f * len(gt_spells))))
        _p = gt_codes.copy()
        for _sp_idx in _perm[:_n_flip]:
            _a, _b, _code = gt_spells[_sp_idx]
            _p[_a:_b] = pm.A if _code == pm.H else pm.H
        _eval_perturbed("5. Flip team on fraction f of spells", f"f = {_f:.2f} ({_n_flip}/17 spells)", _p)

    q02_df = pl.DataFrame(_rows_q02)

    # Q0.3: Published baselines decomposition [GT-5MIN]
    _baseline_names = ["baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"]
    _rows_q03 = []
    for _bname in _baseline_names:
        _bpath = ROOT / "reference" / "evaluation" / "results" / f"{_bname}.csv"
        _bcodes = pm.label_codes(pm.read_label_csv(str(_bpath)))
        _bres = pm.evaluate(gt_codes, _bcodes)
        _rows_q03.append({
            "Baseline": _bname,
            "spellF1 (mean)": round(_bres["spell_f1_mean"], 4),
            "spellF1 @ 0.5": round(_bres["spell_f1_at_headline"], 4),
            "macroF1": round(_bres["macro_f1"], 4),
            "alive_dead_f1": round(_bres["alive_dead_f1"], 4),
            "team_acc_both_alive": round(_bres["team_acc_both_alive"], 4),
            "team_acc_gt_alive": round(_bres["team_acc_gt_alive"], 4),
            "n_gt_spells": _bres["n_gt_spells"],
            "n_pred_spells": _bres["n_pred_spells"],
            "mean_spell_dur_s": round(_bres["mean_spell_duration_s"], 2),
            "fragmentation": round(_bres["fragmentation"], 2),
        })
    q03_df = pl.DataFrame(_rows_q03)

    mo.vstack([
        mo.md("## §0.3 Q0.2 — What does `spellF1` actually reward? `[DERIVED]`\n"
              "**Finding:** **Fragmentation (Distortion 2) is by far the most destructive failure mode**: splitting each spell into just `k=2` pieces crashes `spellF1` from **1.0000 → 0.2549** while `macroF1` stays at **0.9966**. At `k=5`, `spellF1` hits **0.0000** while `macroF1` is still **0.9850**! "
              "By contrast, erasing all 4 dead-ball runs (Distortion 3) leaves `spellF1` at **0.7500**, and a 25-frame (1.0s) uniform lag leaves `spellF1` at **0.6471**."),
        q02_df,
        mo.md("## §0.4 Q0.3 — Which sub-problem binds: alive/dead, or home/away? `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** Across the baseline smoothing sweep (`raw` → `smooth125`), `alive_dead_f1` is completely frozen (**0.8961 → 0.8950**, range 0.0033) and `team_acc_both_alive` moves modestly (**0.7782 → 0.8583**), while `n_pred_spells` drops from **131 → 16** (`fragmentation` **10.05 → 1.05**) and drives the **11.2× jump in `spellF1`** (**0.0378 → 0.4242**)."),
        q03_df,
    ])
    return


@app.cell
def _(FPS, gt_codes, gt_runs, gt_segs, gt_spells, mo, np, pl, pm, win_mask):
    # =========================================================================
    # §2. GROUND-TRUTH ANATOMY [GT-5MIN] & [DERIVED] (Q2.1 - Q2.5)
    # =========================================================================
    _evidence = "GT-5MIN"
    _win_codes = gt_codes[win_mask]

    # Q2.1: Class priors & constant-guess macroF1 floors
    _priors_rows = []
    for _code, _label in [(pm.H, "H (Home / Barcelona)"), (pm.A, "A (Away / Athletic Club)"), (pm.D, "D (Dead ball)")]:
        _cnt = int(np.sum(_win_codes == _code))
        _share = _cnt / len(_win_codes)
        _const_pred = np.full_like(gt_codes, _code)
        _const_eval = pm.evaluate(gt_codes, _const_pred)
        _seg_cnt = sum(1 for _, _, _c in gt_segs if _c == _code)
        _priors_rows.append({
            "Class": _label,
            "Frames": _cnt,
            "Frame Share (%)": round(_share * 100, 2),
            "Segments": _seg_cnt,
            "Segment Share (%)": round(_seg_cnt / len(gt_segs) * 100, 2),
            "Constant-Guess macroF1": round(_const_eval["macro_f1"], 4),
            "Constant-Guess spellF1": round(_const_eval["spell_f1_mean"], 4),
        })
    q21_df = pl.DataFrame(_priors_rows)

    # Q2.2 & Q2.3: All 17 possession spells + boundary tolerances [GT-5MIN] / [DERIVED]
    _spell_rows = []
    for _idx, (_a, _b, _c) in enumerate(gt_spells, start=1):
        _f_len = _b - _a
        _s_len = _f_len / FPS
        # b <= T(1 - t) / (2(1 + t))
        _b_03 = _s_len * (1.0 - 0.3) / (2.0 * (1.0 + 0.3))
        _b_05 = _s_len * (1.0 - 0.5) / (2.0 * (1.0 + 0.5))
        _b_07 = _s_len * (1.0 - 0.7) / (2.0 * (1.0 + 0.7))
        _spell_rows.append({
            "Spell #": _idx,
            "Team": pm.LABELS[_c],
            "Start Frame": _a,
            "End Frame (excl)": _b,
            "Duration (frames)": _f_len,
            "Duration (s)": round(_s_len, 2),
            "Max err b @ IoU 0.3 (s)": round(_b_03, 2),
            "Max err b @ IoU 0.5 (s)": round(_b_05, 2),
            "Max err b @ IoU 0.7 (s)": round(_b_07, 2),
        })
    q22_spells_df = pl.DataFrame(_spell_rows).sort("Duration (s)")

    # Survival table for Q2.3
    _surv_rows = []
    _spell_durs_s = np.array([(_b - _a) / FPS for _a, _b, _ in gt_spells])
    for _iou_t in (0.3, 0.5, 0.7):
        _max_b = _spell_durs_s * (1.0 - _iou_t) / (2.0 * (1.0 + _iou_t))
        _surv_rows.append({
            "IoU Threshold": _iou_t,
            "Survive b = 0.5s (12.5f)": f"{int(np.sum(_max_b >= 0.5))}/17 ({np.mean(_max_b >= 0.5)*100:.1f}%)",
            "Survive b = 1.0s (25f)": f"{int(np.sum(_max_b >= 1.0))}/17 ({np.mean(_max_b >= 1.0)*100:.1f}%)",
            "Survive b = 2.0s (50f)": f"{int(np.sum(_max_b >= 2.0))}/17 ({np.mean(_max_b >= 2.0)*100:.1f}%)",
        })
    q23_survival_df = pl.DataFrame(_surv_rows)

    # Summary quantiles for H/A spells vs D runs
    _d_durs_s = np.array([(_b - _a) / FPS for _a, _b, _c in gt_segs if _c == pm.D])
    def _quant_row(name, arr):
        return {
            "Segment Type": name,
            "Count": len(arr),
            "Min (s)": round(float(np.min(arr)), 2),
            "Q25 (s)": round(float(np.percentile(arr, 25)), 2),
            "Median (s)": round(float(np.median(arr)), 2),
            "Mean (s)": round(float(np.mean(arr)), 2),
            "Q75 (s)": round(float(np.percentile(arr, 75)), 2),
            "Max (s)": round(float(np.max(arr)), 2),
        }
    q22_summary_df = pl.DataFrame([
        _quant_row("Possession Spells (H & A)", _spell_durs_s),
        _quant_row("Home Spells (H)", np.array([(_b - _a) / FPS for _a, _b, _c in gt_spells if _c == pm.H])),
        _quant_row("Away Spells (A)", np.array([(_b - _a) / FPS for _a, _b, _c in gt_spells if _c == pm.A])),
        _quant_row("Dead-Ball Runs (D)", _d_durs_s),
    ])

    # Q2.4: Transition counts & inter-transition gaps
    _gt_trans = pm.transitions(gt_codes, gt_runs)
    _pair_counts = {}
    for _f, _from, _to in _gt_trans:
        _key = f"{pm.LABELS[_from]} → {pm.LABELS[_to]}"
        _pair_counts[_key] = _pair_counts.get(_key, 0) + 1
    _trans_frames = np.array([_t[0] for _t in _gt_trans])
    _inter_gaps_f = np.diff(_trans_frames)
    _inter_gaps_s = _inter_gaps_f / FPS

    q24_df = pl.DataFrame([
        {"Transition Pair": _k, "Count": _v, "Share of 20 Transitions (%)": round(_v / len(_gt_trans) * 100, 1)}
        for _k, _v in sorted(_pair_counts.items(), key=lambda x: -x[1])
    ])

    q24_gaps_df = pl.DataFrame([{
        "Total Transitions": len(_gt_trans),
        "Min Gap (frames / s)": f"{int(_inter_gaps_f.min())}f ({_inter_gaps_s.min():.2f}s)",
        "Q25 Gap (frames / s)": f"{int(np.percentile(_inter_gaps_f, 25))}f ({np.percentile(_inter_gaps_s, 25):.2f}s)",
        "Median Gap (frames / s)": f"{int(np.median(_inter_gaps_f))}f ({np.median(_inter_gaps_s):.2f}s)",
        "Mean Gap (frames / s)": f"{_inter_gaps_f.mean():.1f}f ({_inter_gaps_s.mean():.2f}s)",
        "Max Gap (frames / s)": f"{int(_inter_gaps_f.max())}f ({_inter_gaps_s.max():.2f}s)",
    }])

    # Q2.5: Dead-ball runs anatomy
    _d_rows = []
    for _idx, (_a, _b, _c) in enumerate(gt_segs):
        if _c == pm.D:
            _prev_lbl = pm.LABELS[gt_segs[_idx - 1][2]] if _idx > 0 else "START"
            _next_lbl = pm.LABELS[gt_segs[_idx + 1][2]] if _idx + 1 < len(gt_segs) else "END"
            _d_rows.append({
                "D Run #": len(_d_rows) + 1,
                "Start Frame": _a,
                "End Frame (excl)": _b,
                "Duration (frames)": _b - _a,
                "Duration (s)": round((_b - _a) / FPS, 2),
                "State Before": _prev_lbl,
                "State After": _next_lbl,
                "Team Turnover?": "Yes" if _prev_lbl != _next_lbl else "No (Same Team)",
            })
    q25_df = pl.DataFrame(_d_rows)

    mo.vstack([
        mo.md(f"## §2. Ground-Truth Anatomy `[{_evidence}]` & `[DERIVED]`\n"
              f"**Evidence Tag: `[{_evidence}]`** (17 possessions, 21 segments, 4 dead-ball runs)"),
        mo.md("### Q2.1 — Label distribution & constant-guess floors `[GT-5MIN]`"),
        q21_df,
        mo.md("### Q2.2 — Duration summary & all 17 possession spells + Q2.3 boundary tolerances `[GT-5MIN]` / `[DERIVED]`"),
        q22_summary_df,
        q22_spells_df,
        q23_survival_df,
        mo.md("### Q2.4 — Possession transitions & inter-transition gaps `[GT-5MIN]`\n"
              "**Finding:** Minimum gap between transitions is **19 frames (0.76s)**, followed by **33 frames (1.32s)**. Any smoothing window wider than ~20 frames risks erasing short possessions."),
        q24_df,
        q24_gaps_df,
        mo.md("### Q2.5 — Dead-ball runs in the public window `[GT-5MIN]`"),
        q25_df,
    ])
    return


@app.cell
def _(
    FPS,
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    N_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    ball_present,
    ball_speed_raw,
    ball_x_norm,
    ball_y_norm,
    gt_codes,
    mo,
    np,
    pl,
    pm,
    win_mask,
):
    # =========================================================================
    # §3. THE BALL SIGNAL [FULL] & [GT-5MIN] (Q3.1 - Q3.4)
    # =========================================================================
    # Helper to extract contiguous missing runs (ball_present == False)
    def _missing_runs(present_mask):
        _miss = (~present_mask).astype(np.int8)
        _edges = np.diff(np.concatenate(([0], _miss, [0])))
        _starts = np.flatnonzero(_edges == 1)
        _stops = np.flatnonzero(_edges == -1)
        return _starts, _stops, _stops - _starts

    _starts_all, _stops_all, _lens_all = _missing_runs(ball_present)

    # Classify each gap by where it starts
    def _phase_of_frame(f):
        if f < HALF1_START:
            return "Pre-match (0–11,184)"
        if f <= HALF1_END:
            return "1st Half (11,185–90,651)"
        if f < HALF2_START:
            return "Halftime (90,652–110,807)"
        if f <= HALF2_END:
            return "2nd Half (110,808–190,197)"
        return "Post-match (190,198–199,318)"

    _gap_records = []
    for _s, _e, _l in zip(_starts_all, _stops_all, _lens_all):
        _gap_records.append({
            "start": int(_s),
            "stop": int(_e),
            "length": int(_l),
            "phase": _phase_of_frame(int(_s)),
        })
    _gaps_df = pl.DataFrame(_gap_records)

    q31_phase_df = (
        _gaps_df.group_by("phase")
        .agg([
            pl.len().alias("n_gaps"),
            pl.col("length").sum().alias("missing_frames"),
            pl.col("length").median().round(1).alias("median_gap_frames"),
            (pl.col("length").median() / FPS).round(2).alias("median_gap_s"),
            pl.col("length").quantile(0.90).round(1).alias("p90_gap_frames"),
            pl.col("length").max().alias("max_gap_frames"),
            (pl.col("length").max() / FPS).round(1).alias("max_gap_s"),
        ])
        .sort("missing_frames", descending=True)
    )

    _tot_miss = int(_lens_all.sum())
    _graded_gaps = _gaps_df.filter(pl.col("phase").is_in(["1st Half (11,185–90,651)", "2nd Half (110,808–190,197)"]))["length"].to_numpy()
    _tot_miss_graded = int(_graded_gaps.sum())

    _thresh_rows = []
    for _th in (5, 12, 25, 50, 125):
        _f_all = int(_lens_all[_lens_all <= _th].sum())
        _f_gr = int(_graded_gaps[_graded_gaps <= _th].sum())
        _thresh_rows.append({
            "Max Gap Threshold": f"≤ {_th} frames ({_th/FPS:.2f}s)",
            "All Match Gaps ≤ Thresh": f"{int(np.sum(_lens_all <= _th))}/{len(_lens_all)}",
            "All Match Missing Frames Covered (%)": round(_f_all / _tot_miss * 100, 3),
            "Graded Gaps ≤ Thresh": f"{int(np.sum(_graded_gaps <= _th))}/{len(_graded_gaps)}",
            "Graded Missing Frames Covered (%)": round(_f_gr / _tot_miss_graded * 100, 3),
        })
    q31_thresh_df = pl.DataFrame(_thresh_rows)

    # Q3.2: Ball availability vs GT label [GT-5MIN]
    _win_codes = gt_codes[win_mask]
    _win_bp = ball_present[win_mask]
    _win_bx = ball_x_norm[win_mask]
    _win_by = ball_y_norm[win_mask]

    _q32_rows = []
    for _c, _name in [(pm.H, "H (Barcelona)"), (pm.A, "A (Athletic)"), (pm.D, "D (Dead ball)")]:
        _cmask = _win_codes == _c
        _n_tot = int(_cmask.sum())
        _n_pres = int((_cmask & _win_bp).sum())
        _n_abs = _n_tot - _n_pres
        _q32_rows.append({
            "GT Class": _name,
            "Total Frames": _n_tot,
            "Ball Present Frames": _n_pres,
            "P(Present | Class) (%)": round(_n_pres / _n_tot * 100, 2),
            "P(Class | Present) (%)": round(_n_pres / int(_win_bp.sum()) * 100, 2),
            "Ball Absent Frames": _n_abs,
            "P(Class | Absent) (%)": round(_n_abs / int((~_win_bp).sum()) * 100, 2),
            "Mean Ball X (when present)": round(float(np.nanmean(_win_bx[_cmask & _win_bp])), 3),
            "Mean Ball Y (when present)": round(float(np.nanmean(_win_by[_cmask & _win_bp])), 3),
        })
    q32_df = pl.DataFrame(_q32_rows)

    # Q3.3: Physical plausibility of derived ball speed [FULL]
    _consec_present = ball_present[1:] & ball_present[:-1]
    _dbx_m = np.diff(ball_x_norm) * PITCH_LENGTH
    _dby_m = np.diff(ball_y_norm) * PITCH_WIDTH
    _step_speed_mps = np.full(N_FRAMES, np.nan, dtype=np.float32)
    _step_speed_mps[1:][_consec_present] = np.hypot(_dbx_m[_consec_present], _dby_m[_consec_present]) * FPS

    _valid_speeds = _step_speed_mps[~np.isnan(_step_speed_mps)]
    _fast_40_mask = _step_speed_mps > 40.0
    _fast_50_mask = _step_speed_mps > 50.0

    # Localize >40 m/s steps
    _fx = ball_x_norm[_fast_40_mask]
    _fy = ball_y_norm[_fast_40_mask]
    _near_border = (_fx < 0.05) | (_fx > 0.95) | (_fy < 0.05) | (_fy > 0.95)
    _in_box = ((_fx < 0.16) | (_fx > 0.84)) & (_fy >= 0.21) & (_fy <= 0.79)

    q33_df = pl.DataFrame([{
        "Consecutive Present Steps": len(_valid_speeds),
        "p50 (m/s)": round(float(np.percentile(_valid_speeds, 50)), 2),
        "p90 (m/s)": round(float(np.percentile(_valid_speeds, 90)), 2),
        "p99 (m/s)": round(float(np.percentile(_valid_speeds, 99)), 2),
        "p99.9 (m/s)": round(float(np.percentile(_valid_speeds, 99.9)), 2),
        "Max (m/s)": round(float(np.max(_valid_speeds)), 2),
        "Steps > 40 m/s": f"{int(_fast_40_mask.sum())} ({_fast_40_mask.sum()/len(_valid_speeds)*100:.2f}%)",
        "Steps > 50 m/s": f"{int(_fast_50_mask.sum())} ({_fast_50_mask.sum()/len(_valid_speeds)*100:.2f}%)",
        ">40 m/s Near Pitch Border (<5%)": f"{int(_near_border.sum())} ({_near_border.mean()*100:.1f}%)",
        ">40 m/s Inside Penalty Boxes": f"{int(_in_box.sum())} ({_in_box.mean()*100:.1f}%)",
    }])

    # Q3.4: Exported ball_speed audit [FULL]
    _bs_valid = ~np.isnan(ball_speed_raw)
    _bs_frames = np.flatnonzero(_bs_valid)
    _bs_in_win = int(_bs_valid[win_mask].sum())
    assert _bs_in_win == 0, "Expected 0 exported ball_speed values in public window"
    _both_speed = _bs_valid & (~np.isnan(_step_speed_mps))
    _speed_corr = float(np.corrcoef(ball_speed_raw[_both_speed], _step_speed_mps[_both_speed])[0, 1])

    q34_df = pl.DataFrame([{
        "Non-NaN Count": f"{int(_bs_valid.sum())} / {N_FRAMES} ({_bs_valid.mean()*100:.2f}%)",
        "Frame Range": f"{int(_bs_frames[0])} .. {int(_bs_frames[-1])}",
        "In Public 5-min Window": _bs_in_win,
        "Present when ball_x is NaN": int((_bs_valid & (~ball_present)).sum()),
        "Corr with Derived Speed": round(_speed_corr, 4),
        "Decision": "DROP (0% coverage in validation window)",
    }])

    mo.vstack([
        mo.md("## §3. The Ball Signal `[FULL]` & `[GT-5MIN]`"),
        mo.md("### Q3.1 — Missing ball gaps by phase & interpolation ceiling `[FULL]`\n"
              "**Finding:** Missingness is blocky (median **230 frames / 9.2s**; **180 of 186** gaps occur during 1st/2nd half graded play, while halftime/pre-match/post-match account for the massive multi-minute gaps). Even inside graded play, gaps ≤12 frames account for only **0.11%** of missing frames — **do not build an interpolation module**."),
        q31_phase_df,
        q31_thresh_df,
        mo.md("### Q3.2 — Match state conditioned on ball presence `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** When the ball is **absent** (885 frames in window), the true state is **55.8% `D`, 43.7% `A`, and 0.5% `H`**. Calling every ball-absent frame `D` is wrong **44.2%** of the time in the window and disproportionately erases Away (`A`) possessions."),
        q32_df,
        mo.md("### Q3.3 — Physical plausibility of derived ball speed `[FULL]`\n"
              "**Finding:** **2.15%** of consecutive ball steps (2,437 frames) exceed **40 m/s** (and **0.98%** exceed **50 m/s**), crossing the 2% decision threshold — add a ball-speed plausibility gate."),
        q33_df,
        mo.md("### Q3.4 — Audit of exported `ball_speed` column `[FULL]`"),
        q34_df,
    ])
    return


@app.cell
def _(
    PITCH_LENGTH,
    PITCH_WIDTH,
    ball_present,
    d_A,
    d_H,
    dist_m,
    graded_mask,
    gt_codes,
    mo,
    np,
    pl,
    pm,
    px_norm,
    py_norm,
    slot_is_home,
    slots_df,
    win_mask,
):
    # =========================================================================
    # §4. PLAYER-BALL GEOMETRY & TEAM ATTRIBUTION [FULL] & [GT-5MIN] (Q4.1 - Q4.3)
    # =========================================================================
    # Q4.1: Track slot reliability and duplicate detection (<1.0m same-team overlap) [FULL]
    _slot_active_counts = np.sum(~np.isnan(px_norm), axis=0)
    _slot_graded_counts = np.sum(~np.isnan(px_norm[graded_mask]), axis=0)

    # Check pairwise distances within same team on graded frames
    _dup_counts_per_slot = np.zeros(46, dtype=np.int64)
    _px_m = px_norm * PITCH_LENGTH
    _py_m = py_norm * PITCH_WIDTH

    for _i in range(46):
        for _j in range(_i + 1, 46):
            if slot_is_home[_i] != slot_is_home[_j]:
                continue
            _d_ij = np.hypot(_px_m[:, _i] - _px_m[:, _j], _py_m[:, _i] - _py_m[:, _j])
            _is_dup = _d_ij < 1.0
            _dup_counts_per_slot[_i] += int(np.sum(_is_dup))
            _dup_counts_per_slot[_j] += int(np.sum(_is_dup))

    _dup_rate_pct = np.where(_slot_active_counts > 0, _dup_counts_per_slot / _slot_active_counts * 100.0, 0.0)

    # Nearest slot overall on ball-present frames
    _dist_clean = np.where(np.isnan(dist_m), np.inf, dist_m)
    _nearest_slot_idx = np.argmin(_dist_clean, axis=1)

    q41_slots_df = slots_df.with_columns([
        pl.Series("active_frames_all", _slot_active_counts),
        pl.Series("active_frames_graded", _slot_graded_counts),
        pl.Series("dup_frames_<1m", _dup_counts_per_slot),
        pl.Series("dup_rate_pct", np.round(_dup_rate_pct, 2)),
    ]).sort("dup_rate_pct", descending=True)

    _high_dup_slots = q41_slots_df.filter(pl.col("dup_rate_pct") > 5.0)
    _short_lived_slots = q41_slots_df.filter(pl.col("active_frames_all") < 25)

    # Q4.2: Class-conditional distance-to-ball (d_H, d_A, d_A - d_H) [GT-5MIN]
    _win_bp_mask = win_mask & ball_present
    _win_codes_bp = gt_codes[_win_bp_mask]
    _dH_bp = d_H[_win_bp_mask]
    _dA_bp = d_A[_win_bp_mask]
    _margin_bp = _dA_bp - _dH_bp  # positive => Home is closer

    def _dist_summary_row(label_name, mask_sub):
        _h = _dH_bp[mask_sub]
        _a = _dA_bp[mask_sub]
        _m = _margin_bp[mask_sub]
        _min_both = np.minimum(_h, _a)
        return {
            "GT Class (ball present)": label_name,
            "Frames": int(mask_sub.sum()),
            "d_H p25 / Med / p75 (m)": f"{np.percentile(_h,25):.2f} / {np.median(_h):.2f} / {np.percentile(_h,75):.2f}",
            "d_A p25 / Med / p75 (m)": f"{np.percentile(_a,25):.2f} / {np.median(_a):.2f} / {np.percentile(_a,75):.2f}",
            "min(d_H,d_A) Med / p90 (m)": f"{np.median(_min_both):.2f} / {np.percentile(_min_both,90):.2f}",
            "Margin (d_A - d_H) p25 / Med / p75 (m)": f"{np.percentile(_m,25):+.2f} / {np.median(_m):+.2f} / {np.percentile(_m,75):+.2f}",
        }

    q42_df = pl.DataFrame([
        _dist_summary_row("All Window Ball-Present", np.ones_like(_win_codes_bp, dtype=bool)),
        _dist_summary_row("H (Barcelona Possession)", _win_codes_bp == pm.H),
        _dist_summary_row("A (Athletic Possession)", _win_codes_bp == pm.A),
        _dist_summary_row("D (Dead Ball)", _win_codes_bp == pm.D),
    ])

    # Q4.3: Nearest-player accuracy & margin buckets & flicker count [GT-5MIN]
    _alive_bp = (_win_codes_bp == pm.H) | (_win_codes_bp == pm.A)
    _gt_alive_bp = _win_codes_bp[_alive_bp]
    _pred_nearest_bp = np.where(_dH_bp[_alive_bp] <= _dA_bp[_alive_bp], pm.H, pm.A)
    _abs_margin_alive = np.abs(_margin_bp[_alive_bp])
    _correct_alive = _pred_nearest_bp == _gt_alive_bp

    _b1 = _abs_margin_alive <= 0.5
    _b2 = (_abs_margin_alive > 0.5) & (_abs_margin_alive <= 2.0)
    _b3 = _abs_margin_alive > 2.0

    # Flicker over all 7,500 window frames (using nearest when present, D when absent)
    _raw_win_codes = np.full(int(win_mask.sum()), pm.D, dtype=np.int8)
    _w_bp = ball_present[win_mask]
    _raw_win_codes[_w_bp] = np.where(d_H[win_mask][_w_bp] <= d_A[win_mask][_w_bp], pm.H, pm.A)
    _n_raw_changes_all = int(np.sum(_raw_win_codes[1:] != _raw_win_codes[:-1]))
    _w_bp_team = np.where(d_H[win_mask][_w_bp] <= d_A[win_mask][_w_bp], pm.H, pm.A)
    _n_team_flips_bp = int(np.sum(_w_bp_team[1:] != _w_bp_team[:-1]))

    q43_df = pl.DataFrame([
        {"Slice": "All Alive Ball-Present Frames", "Frames": int(_alive_bp.sum()), "Share (%)": 100.0, "Nearest-Team Accuracy (%)": round(float(_correct_alive.mean() * 100), 2)},
        {"Slice": "GT == H (Home Alive)", "Frames": int((_gt_alive_bp == pm.H).sum()), "Share (%)": round(float((_gt_alive_bp == pm.H).mean() * 100), 1), "Nearest-Team Accuracy (%)": round(float(_correct_alive[_gt_alive_bp == pm.H].mean() * 100), 2)},
        {"Slice": "GT == A (Away Alive)", "Frames": int((_gt_alive_bp == pm.A).sum()), "Share (%)": round(float((_gt_alive_bp == pm.A).mean() * 100), 1), "Nearest-Team Accuracy (%)": round(float(_correct_alive[_gt_alive_bp == pm.A].mean() * 100), 2)},
        {"Slice": "Margin |d_A - d_H| ≤ 0.5 m (Contested)", "Frames": int(_b1.sum()), "Share (%)": round(float(_b1.mean() * 100), 1), "Nearest-Team Accuracy (%)": round(float(_correct_alive[_b1].mean() * 100), 2)},
        {"Slice": "Margin 0.5 < |d_A - d_H| ≤ 2.0 m", "Frames": int(_b2.sum()), "Share (%)": round(float(_b2.mean() * 100), 1), "Nearest-Team Accuracy (%)": round(float(_correct_alive[_b2].mean() * 100), 2)},
        {"Slice": "Margin |d_A - d_H| > 2.0 m (Clear Control)", "Frames": int(_b3.sum()), "Share (%)": round(float(_b3.mean() * 100), 1), "Nearest-Team Accuracy (%)": round(float(_correct_alive[_b3].mean() * 100), 2)},
    ])

    mo.vstack([
        mo.md("## §4. Player-Ball Geometry and Team Attribution `[FULL]` & `[GT-5MIN]`"),
        mo.md(f"### Q4.1 — Track slot audit & duplicate overlap (<1.0m same-team) `[FULL]`\n"
              f"**Finding:** Across all 46 slots, **{len(_short_lived_slots)} slots** have `< 25` active frames, and **{len(_high_dup_slots)} slots** exceed the 5% duplicate-overlap threshold (max duplicate rate is **{q41_slots_df['dup_rate_pct'].max():.2f}%**). All 46 slots are clean enough to retain."),
        q41_slots_df.head(10),
        mo.md("### Q4.2 — Distance-to-ball distribution by GT class `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** When `H` has possession, median `d_H = 1.66 m` vs `d_A = 6.91 m` (median margin `+4.66 m`). When `A` has possession, median `d_A = 1.94 m` vs `d_H = 6.13 m` (median margin `-3.65 m`). During `D` (ball present), the ball is still near players (`min(d_H, d_A)` median `1.97 m`), so absolute distance alone does not separate `D` from live play — relative margin `d_A - d_H` separates `H` vs `A`."),
        q42_df,
        mo.md(f"### Q4.3 — Nearest-player accuracy & flicker rate `[GT-5MIN]`\n"
              f"**Evidence Tag: `[GT-5MIN]`**\n"
              f"**Finding:** Per-frame accuracy on alive ball-present frames is **77.82%** (**90.69%** when `|margin| > 2m`, **56.90%** when `|margin| ≤ 0.5m`). "
              f"However, the raw nearest-team label flips **{_n_team_flips_bp} times** across ball-present frames (and **{_n_raw_changes_all} transitions** with `D` gaps vs **20 true transitions**)."),
        q43_df,
    ])
    return


@app.cell
def _(
    FPS,
    N_FRAMES,
    ROOT,
    WIN_START,
    ball_present,
    d_A,
    d_H,
    gt_codes,
    mo,
    np,
    pl,
    pm,
    win_mask,
):
    # =========================================================================
    # §5. TEMPORAL FRAGMENTATION — HIGHEST PRIORITY [GT-5MIN] & [DERIVED] (Q5.1 - Q5.4)
    # =========================================================================
    # Build full-match raw nearest code array: 0=H, 1=A, 2=D (when ball missing)
    _raw_full = np.full(N_FRAMES, pm.D, dtype=np.int8)
    _raw_full[ball_present] = np.where(d_H[ball_present] <= d_A[ball_present], pm.H, pm.A)

    # Q5.1: Run-length distribution of raw nearest-team signal in window [GT-5MIN]
    _w_bp = ball_present[win_mask]
    _w_team_bp = _raw_full[win_mask][_w_bp]
    _breaks_bp = np.flatnonzero(_w_team_bp[1:] != _w_team_bp[:-1]) + 1
    _bounds_bp = np.concatenate(([0], _breaks_bp, [len(_w_team_bp)]))
    _runs_bp_len = np.diff(_bounds_bp)

    # Carry-forward team rule across ball-absent frames in window
    _w_carry = _raw_full[win_mask].copy()
    _last_t = pm.H
    for _i in range(len(_w_carry)):
        if _w_carry[_i] in (pm.H, pm.A):
            _last_t = _w_carry[_i]
        else:
            _w_carry[_i] = _last_t
    _breaks_cf = np.flatnonzero(_w_carry[1:] != _w_carry[:-1]) + 1
    _runs_cf_len = np.diff(np.concatenate(([0], _breaks_cf, [len(_w_carry)])))

    def _run_stats_row(name, rlens):
        return {
            "Signal Rule": name,
            "Total Runs": len(rlens),
            "p10 (frames / s)": f"{int(np.percentile(rlens,10))}f ({np.percentile(rlens,10)/FPS:.2f}s)",
            "Median (frames / s)": f"{int(np.median(rlens))}f ({np.median(rlens)/FPS:.2f}s)",
            "p90 (frames / s)": f"{int(np.percentile(rlens,90))}f ({np.percentile(rlens,90)/FPS:.2f}s)",
            "Runs < 6f (0.24s) (%)": round(float(np.mean(rlens < 6) * 100), 1),
            "Runs < 12f (0.48s) (%)": round(float(np.mean(rlens < 12) * 100), 1),
            "Runs < 25f (1.0s) (%)": round(float(np.mean(rlens < 25) * 100), 1),
            "Runs < 50f (2.0s) (%)": round(float(np.mean(rlens < 50) * 100), 1),
        }

    q51_df = pl.DataFrame([
        _run_stats_row("Raw Nearest Team (ball-present frames only)", _runs_bp_len),
        _run_stats_row("Last-Team Carry-Forward (across missing ball)", _runs_cf_len),
    ])

    # Q5.2: Sweep smoothing window & plot/tabulate spellF1 vs n_pred_spells [DERIVED]
    def _smooth_codes(raw_c, half_w):
        if half_w <= 0:
            return raw_c.copy()
        _hc = np.concatenate(([0], np.cumsum(raw_c == pm.H)))
        _ac = np.concatenate(([0], np.cumsum(raw_c == pm.A)))
        _idx = np.arange(len(raw_c))
        _lo = np.maximum(0, _idx - half_w)
        _hi = np.minimum(len(raw_c), _idx + half_w + 1)
        _h_sum = _hc[_hi] - _hc[_lo]
        _a_sum = _ac[_hi] - _ac[_lo]
        _out = np.full(len(raw_c), pm.D, dtype=np.int8)
        _seen = (_h_sum + _a_sum) > 0
        _out[_seen] = np.where(_h_sum[_seen] >= _a_sum[_seen], pm.H, pm.A)
        return _out

    _graded_runs_full = [(11185, 90652), (110808, 190198)]
    _n_graded_5min_blocks = 158857.0 / 7500.0  # 21.181 five-minute blocks in full graded match

    _q52_rows = []
    for _w in (0, 3, 6, 12, 18, 25, 37, 50, 62, 75, 100, 125, 150, 200):
        _sm = _smooth_codes(_raw_full, _w)
        _ev = pm.evaluate(gt_codes, _sm)
        _full_spells = pm.possession_spells(_sm, _graded_runs_full)
        _full_mean_dur = float(np.mean([b - a for a, b, _ in _full_spells]) / FPS) if _full_spells else 0.0
        _q52_rows.append({
            "Half-Window w (frames)": _w,
            "Window Width (s)": round((2 * _w + 1) / FPS, 2),
            "Win n_pred_spells (GT=17)": _ev["n_pred_spells"],
            "Win n_pred / n_gt": round(_ev["n_pred_spells"] / 17.0, 2),
            "Win Mean Dur (s)": round(_ev["mean_spell_duration_s"], 2),
            "Full Match Spells / 5min": round(len(_full_spells) / _n_graded_5min_blocks, 1),
            "Full Match Mean Dur (s)": round(_full_mean_dur, 2),
            "Win spellF1 (mean)": round(_ev["spell_f1_mean"], 4),
            "Win macroF1": round(_ev["macro_f1"], 4),
        })
    q52_df = pl.DataFrame(_q52_rows)

    # Q5.3: Transition diagnostics at tau = 12 and 25 frames [GT-5MIN]
    _q53_rows = []
    for _bname in ["baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"]:
        _bpath = ROOT / "reference" / "evaluation" / "results" / f"{_bname}.csv"
        _bcodes = pm.label_codes(pm.read_label_csv(str(_bpath)))
        _bev = pm.evaluate(gt_codes, _bcodes)
        for _tau in (12, 25):
            _tr = _bev["transitions"][_tau]
            _q53_rows.append({
                "Submission": _bname,
                "τ (frames)": _tau,
                "Precision": round(_tr["precision"], 3),
                "Recall": round(_tr["recall"], 3),
                "Trans F1": round(_tr["f1"], 3),
                "Matched / GT": f"{_tr['n_matched']}/{_tr['n_gt']}",
                "Missed": _tr["n_missed"],
                "Spurious": _tr["n_spurious"],
                "Ambiguous": _tr["n_ambiguous"],
                "Mean Signed Latency (f)": round(_tr["mean_latency"], 2) if _tr["mean_latency"] is not None else None,
                "Median Signed Latency (f)": round(_tr["median_latency"], 2) if _tr["median_latency"] is not None else None,
            })
    q53_df = pl.DataFrame(_q53_rows)

    # Q5.4: Cost of abstention (X) [DERIVED]
    _base50 = _smooth_codes(_raw_full, 50)
    _base50_ev = pm.evaluate(gt_codes, _base50)
    _gt_ev = pm.evaluate(gt_codes, gt_codes)

    _q54_rows = []
    def _test_abstain(base_name, base_arr, base_res, loc_desc, start_f, length_f):
        _mod = base_arr.copy()
        _mod[start_f : start_f + length_f] = pm.X
        _res = pm.evaluate(gt_codes, _mod)
        _q54_rows.append({
            "Base Signal": base_name,
            "Abstention Location": loc_desc,
            "X Run Length": f"{length_f}f ({length_f/FPS:.2f}s)",
            "Δ spellF1": round(_res["spell_f1_mean"] - base_res["spell_f1_mean"], 4),
            "Δ macroF1": round(_res["macro_f1"] - base_res["macro_f1"], 4),
            "Result spellF1": round(_res["spell_f1_mean"], 4),
            "Result macroF1": round(_res["macro_f1"], 4),
            "n_pred_spells": _res["n_pred_spells"],
        })

    # Mid-possession (inside long H spell 13478..14371)
    # True transition (around 12611 A->H)
    # Inside D run (around 16000)
    # Start of half (11185)
    for _L in (5, 25, 50):
        _test_abstain("GT Oracle", gt_codes, _gt_ev, "1. Inside possession (mid-spell)", 13800, _L)
        _test_abstain("GT Oracle", gt_codes, _gt_ev, "2. Straddling true transition", 12611 - _L // 2, _L)
        _test_abstain("GT Oracle", gt_codes, _gt_ev, "3. Inside dead-ball (D) run", 16000, _L)
        _test_abstain("GT Oracle", gt_codes, _gt_ev, "4. Very start of half (back-filled)", WIN_START, _L)
        _test_abstain("baseline_smooth50", _base50, _base50_ev, "1. Inside possession (mid-spell)", 13800, _L)
        _test_abstain("baseline_smooth50", _base50, _base50_ev, "2. Straddling true transition", 12611 - _L // 2, _L)

    q54_df = pl.DataFrame(_q54_rows)

    mo.vstack([
        mo.md("## §5. Temporal Fragmentation — Highest Priority `[GT-5MIN]` & `[DERIVED]`"),
        mo.md("### Q5.1 — Flicker run-length distribution of raw proximity signal `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** **41.6%** of raw nearest-team runs are shorter than **12 frames (0.48s)**, and **58.4%** are shorter than **25 frames (1.0s)** (median run length is just **17 frames / 0.68s** vs the true median spell of **7.56s**). A minimum-dwell / temporal filter is mandatory."),
        q51_df,
        mo.md("### Q5.2 — `spellF1` vs predicted possession count across smoothing sweep `[DERIVED]`\n"
              "**Finding:** Note why `smooth125` wins on the 5-min window (16 spells, 18.5s mean duration) but loses on the full match: on the full match (59.3% ball visibility vs 88.2% in the window), `w=50` already produces **15.8 spells / 5min** (mean duration **14.33s**), matching the true match scale (~17 spells / 5min, 14.2s), whereas `w=125` over-smooths the full match down to **9.6 spells / 5min** (mean duration **22.88s**)!"),
        q52_df,
        mo.md("### Q5.3 — Transition precision, recall, and signed latency `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**"),
        q53_df,
        mo.md("### Q5.4 — Exact cost of abstention (`X`) `[DERIVED]`\n"
              "**Finding:** Because `fill_abstentions()` forward-fills `X` (and back-fills at the start of a half) before computing `spellF1`, **emitting `X` inside a possession or inside a `D` run or at the start of a half has `ΔspellF1 = 0.0000`** (while costing ~0.001–0.004 in `macroF1`). Across a transition, `X` shifts the transition boundary by `L/2` frames."),
        q54_df,
    ])
    return


@app.cell
def _(
    FPS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    ball_present,
    ball_x_norm,
    ball_y_norm,
    gt_codes,
    gt_segs,
    mo,
    np,
    pl,
    pm,
    ps_raw,
    slot_is_away,
    slot_is_home,
    win_mask,
):
    # =========================================================================
    # §6. DEAD-BALL SIGNATURES [GT-5MIN] (Q6.1 - Q6.2)
    # =========================================================================
    _evidence = "GT-5MIN"
    _win_bp = win_mask & ball_present
    _w_codes = gt_codes[win_mask]
    _w_bp_Local = ball_present[win_mask]
    _w_bx = ball_x_norm[win_mask]
    _w_by = ball_y_norm[win_mask]

    # Q6.1: Out-of-bounds precision & recall at margins + 5-second horizon + lead/lag
    _q61_rows = []
    for _eps, _label in [
        (0.0, "Strict OOB (< 0.0 or > 1.0)"),
        (0.005, "Within 0.5% inside touchline (< 0.005 or > 0.995)"),
        (0.01, "Within 1.0% inside touchline (< 0.01 or > 0.99)"),
        (0.02, "Within 2.0% inside touchline (< 0.02 or > 0.98)"),
    ]:
        _oob_flag = _w_bp_Local & ((_w_bx < _eps) | (_w_bx > 1.0 - _eps) | (_w_by < _eps) | (_w_by > 1.0 - _eps))
        _tp = int(np.sum(_oob_flag & (_w_codes == pm.D)))
        _fp = int(np.sum(_oob_flag & (_w_codes != pm.D)))
        _fn_bp = int(np.sum((~_oob_flag) & _w_bp_Local & (_w_codes == pm.D)))
        _fn_all = int(np.sum((~_oob_flag) & (_w_codes == pm.D)))
        _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _rec_bp = _tp / (_tp + _fn_bp) if (_tp + _fn_bp) else 0.0
        _rec_all = _tp / (_tp + _fn_all) if (_tp + _fn_all) else 0.0

        # 5-second forward horizon: if OOB fires at frame t, does D occur in [t, t+125]?
        _oob_idx = np.flatnonzero(_oob_flag)
        _hit_5s = sum(1 for _idx in _oob_idx if np.any(_w_codes[_idx : min(len(_w_codes), _idx + 126)] == pm.D))
        _prec_5s = _hit_5s / len(_oob_idx) if len(_oob_idx) else 0.0

        _q61_rows.append({
            "Boundary Rule": _label,
            "OOB Frames": int(_oob_flag.sum()),
            "Instant Precision (OOB ⇒ D)": round(_prec * 100, 1),
            "5s-Horizon Precision (D within +5s)": round(_prec_5s * 100, 1),
            "Recall on Ball-Present D (%)": round(_rec_bp * 100, 1),
            "Recall on All D Frames (%)": round(_rec_all * 100, 1),
        })
    q61_df = pl.DataFrame(_q61_rows)

    # Check lead/lag of OOB around the 4 D runs
    _oob_strict_full = ball_present & ((ball_x_norm < 0.0) | (ball_x_norm > 1.0) | (ball_y_norm < 0.0) | (ball_y_norm > 1.0))
    _d_oob_details = []
    for _idx, (_a, _b, _c) in enumerate(gt_segs):
        if _c == pm.D:
            _win_search = np.flatnonzero(_oob_strict_full[_a - 50 : _b + 50]) + (_a - 50)
            _first_oob = int(_win_search[0]) if len(_win_search) else None
            _d_oob_details.append({
                "D Run": f"{_a}..{_b}",
                "OOB Frames Inside D": int(_oob_strict_full[_a:_b].sum()),
                "First OOB Near Run (±50f)": _first_oob,
                "Lead/Lag vs D Start (frames)": (_first_oob - _a) if _first_oob is not None else None,
            })
    q61_leadlag_df = pl.DataFrame(_d_oob_details)

    # Q6.2: Player speeds & derived ball speed during H / A / D [GT-5MIN]
    with np.errstate(all="ignore"):
        _mean_ps = np.nanmean(ps_raw[win_mask], axis=1)
        _p90_ps = np.nanpercentile(ps_raw[win_mask], 90, axis=1)
        _mean_ps_H = np.nanmean(ps_raw[win_mask][:, slot_is_home], axis=1)
        _mean_ps_A = np.nanmean(ps_raw[win_mask][:, slot_is_away], axis=1)

    _consec_bp_w = _w_bp_Local[1:] & _w_bp_Local[:-1]
    _bs_w = np.full(len(_w_codes), np.nan, dtype=np.float32)
    _bs_w[1:][_consec_bp_w] = np.hypot(np.diff(_w_bx)[_consec_bp_w] * PITCH_LENGTH, np.diff(_w_by)[_consec_bp_w] * PITCH_WIDTH) * FPS

    _q62_rows = []
    for _c, _name in [(pm.H, "H (Home Alive)"), (pm.A, "A (Away Alive)"), (pm.D, "D (Dead Ball)")]:
        _m = _w_codes == _c
        _q62_rows.append({
            "GT Class": _name,
            "Frames": int(_m.sum()),
            "Mean Player Speed (m/s)": round(float(np.nanmean(_mean_ps[_m])), 2),
            "p90 Player Speed (m/s)": round(float(np.nanmean(_p90_ps[_m])), 2),
            "Home Team Mean Speed (m/s)": round(float(np.nanmean(_mean_ps_H[_m])), 2),
            "Away Team Mean Speed (m/s)": round(float(np.nanmean(_mean_ps_A[_m])), 2),
            "Derived Ball Speed Med / Mean (m/s)": f"{np.nanmedian(_bs_w[_m]):.2f} / {np.nanmean(_bs_w[_m]):.2f}",
        })
    q62_df = pl.DataFrame(_q62_rows)

    # Best single threshold on mean_player_speed for predicting D vs base rate (19.6%)
    _best_f1, _best_th, _best_prec, _best_rec = 0.0, 0.0, 0.0, 0.0
    _is_d = _w_codes == pm.D
    for _th in np.linspace(1.0, 4.0, 61):
        _pred_d = _mean_ps < _th
        _tp = int(np.sum(_pred_d & _is_d))
        _fp = int(np.sum(_pred_d & (~_is_d)))
        _fn = int(np.sum((~_pred_d) & _is_d))
        _pr = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _rc = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        _f1 = 2 * _pr * _rc / (_pr + _rc) if (_pr + _rc) else 0.0
        if _f1 > _best_f1:
            _best_f1, _best_th, _best_prec, _best_rec = _f1, float(_th), _pr, _rc

    q62_thresh_df = pl.DataFrame([{
        "Feature": "mean_player_speed < threshold",
        "Best Threshold (m/s)": round(_best_th, 2),
        "D Base Rate (%)": 19.59,
        "Precision (%)": round(_best_prec * 100, 1),
        "Recall (%)": round(_best_rec * 100, 1),
        "Best Single-Feature D F1": round(_best_f1, 4),
    }])

    mo.vstack([
        mo.md(f"## §6. Dead-Ball Signatures `[{_evidence}]`\n**Evidence Tag: `[{_evidence}]`**"),
        mo.md("### Q6.1 — Out-of-bounds precision, recall & lead/lag `[GT-5MIN]`\n"
              "**Finding:** Strict OOB has **97.2% instantaneous precision** (**99.1%** within a +5s horizon) for `D`, though it only fires on 1 out of 4 `D` runs (7.1% overall `D` recall). It is a high-precision rule trigger."),
        q61_df,
        q61_leadlag_df,
        mo.md("### Q6.2 — Collective player speeds & ball speed during dead balls `[GT-5MIN]`\n"
              "**Finding:** Mean player speed drops from **2.21 m/s (`H`) / 2.28 m/s (`A`) → 1.43 m/s (`D`)** (and p90 speed drops from **~4.22 m/s → 2.86 m/s**). "
              f"A single threshold (`mean_player_speed < {_best_th:.2f} m/s`) achieves **{ _best_prec*100:.1f}% precision and {_best_rec*100:.1f}% recall (F1 = {_best_f1:.3f})** — 3× the 19.6% base rate and completely independent of ball visibility!"),
        q62_df,
        q62_thresh_df,
    ])
    return


@app.cell
def _(
    N_FRAMES,
    WIN_END,
    WIN_START,
    graded_mask,
    gt_codes,
    gt_runs,
    mo,
    np,
    pl,
    pm,
    tagging_df,
):
    # =========================================================================
    # §7. SMART TAGGING PRIORS [FULL] & [GT-5MIN] (Q7.1 - Q7.4)
    # =========================================================================
    # Q7.1: Coverage & concurrency across all 199,319 frames and 158,857 graded frames [FULL]
    _tag_concurrency = np.zeros(N_FRAMES, dtype=np.int16)
    _code_frame_masks = {}

    for _row in tagging_df.iter_rows(named=True):
        _s = max(0, int(_row["start_frame"]))
        _e = min(N_FRAMES, int(_row["end_frame"]))
        _c = _row["code"]
        _tag_concurrency[_s:_e] += 1
        if _c not in _code_frame_masks:
            _code_frame_masks[_c] = np.zeros(N_FRAMES, dtype=bool)
        _code_frame_masks[_c][_s:_e] = True

    _code_rows = []
    _inst_counts = tagging_df.group_by("code").len().to_dict(as_series=False)
    _inst_map = dict(zip(_inst_counts["code"], _inst_counts["len"]))

    for _c, _cmask in _code_frame_masks.items():
        _f_all = int(_cmask.sum())
        _f_gr = int(_cmask[graded_mask].sum())
        _code_rows.append({
            "Code": _c,
            "Instances": _inst_map.get(_c, 0),
            "Total Frames": _f_all,
            "Share of Video (%)": round(_f_all / N_FRAMES * 100, 2),
            "Graded Frames": _f_gr,
            "Share of Graded (%)": round(_f_gr / int(graded_mask.sum()) * 100, 2),
        })
    q71_codes_df = pl.DataFrame(_code_rows).sort("Graded Frames", descending=True)

    _conc_rows = []
    for _k in range(int(_tag_concurrency.max()) + 1):
        _cnt_all = int(np.sum(_tag_concurrency == _k))
        _cnt_gr = int(np.sum(_tag_concurrency[graded_mask] == _k))
        _conc_rows.append({
            "Simultaneous Active Tags": _k,
            "Video Frames": _cnt_all,
            "Video Share (%)": round(_cnt_all / N_FRAMES * 100, 2),
            "Graded Frames": _cnt_gr,
            "Graded Share (%)": round(_cnt_gr / int(graded_mask.sum()) * 100, 2),
        })
    q71_conc_df = pl.DataFrame(_conc_rows)

    # Q7.2: SET PIECES boundary agreement [GT-5MIN]
    _gt_trans = pm.transitions(gt_codes, gt_runs)
    _d_to_alive_frames = [t[0] for t in _gt_trans if t[1] == pm.D and t[2] in (pm.H, pm.A)] + [WIN_START]
    _alive_to_d_frames = [t[0] for t in _gt_trans if t[1] in (pm.H, pm.A) and t[2] == pm.D]

    _sp_win = tagging_df.filter(
        (pl.col("code") == "SET PIECES") & (pl.col("end_frame") >= WIN_START) & (pl.col("start_frame") <= WIN_END)
    )
    _q72_rows = []
    for _r in _sp_win.iter_rows(named=True):
        _sf, _ef = int(_r["start_frame"]), int(_r["end_frame"])
        _nearest_restart = min(_d_to_alive_frames, key=lambda x: abs(x - _sf))
        _nearest_stoppage = min(_alive_to_d_frames, key=lambda x: abs(x - _ef)) if _alive_to_d_frames else None
        _q72_rows.append({
            "Set Piece Type": _r["type"],
            "Team Tag": _r["team"],
            "Tag Interval": f"{_sf}..{_ef} ({_ef - _sf}f)",
            "Nearest D→Alive Restart": _nearest_restart,
            "Start Offset (sf - restart)": _sf - _nearest_restart,
            "End Offset (ef - restart)": _ef - _nearest_restart,
            "Nearest Alive→D Stoppage": _nearest_stoppage,
        })
    q72_df = pl.DataFrame(_q72_rows)

    # Q7.3: On-ball vs Off-ball implied possessor agreement [GT-5MIN]
    _defensive_codes = {
        "DEFENDING IN DEFENSIVE THIRD",
        "DEFENDING IN MIDDLE THIRD",
        "DEFENDING IN ATTACKING THIRD",
        "DEFENSIVE TRANSITION",
    }

    _win_tag_stats = {}
    for _r in tagging_df.iter_rows(named=True):
        _s = max(WIN_START, int(_r["start_frame"]))
        _e = min(WIN_END + 1, int(_r["end_frame"]))
        if _s >= _e:
            continue
        _c = _r["code"]
        _t = _r["team"]
        if _t not in ("Barcelona", "Atletic Club"):
            continue
        _tag_team_code = pm.H if _t == "Barcelona" else pm.A
        _opp_team_code = pm.A if _tag_team_code == pm.H else pm.H
        _is_def = _c in _defensive_codes
        _implied_code = _opp_team_code if _is_def else _tag_team_code

        _slice_gt = gt_codes[_s:_e]
        _alive_mask = (_slice_gt == pm.H) | (_slice_gt == pm.A)

        if _c not in _win_tag_stats:
            _win_tag_stats[_c] = {"all_f": 0, "raw_hit": 0, "imp_hit": 0, "alive_f": 0, "imp_alive_hit": 0}
        _win_tag_stats[_c]["all_f"] += len(_slice_gt)
        _win_tag_stats[_c]["raw_hit"] += int(np.sum(_slice_gt == _tag_team_code))
        _win_tag_stats[_c]["imp_hit"] += int(np.sum(_slice_gt == _implied_code))
        _win_tag_stats[_c]["alive_f"] += int(_alive_mask.sum())
        _win_tag_stats[_c]["imp_alive_hit"] += int(np.sum(_slice_gt[_alive_mask] == _implied_code))

    _q73_rows = []
    for _c, _st in sorted(_win_tag_stats.items(), key=lambda x: -x[1]["all_f"]):
        _role = "Off-Ball (Defensive → Flip Team)" if _c in _defensive_codes else "On-Ball (Offensive → Keep Team)"
        _imp_all = _st["imp_hit"] / _st["all_f"] * 100 if _st["all_f"] else 0.0
        _imp_alive = _st["imp_alive_hit"] / _st["alive_f"] * 100 if _st["alive_f"] else 0.0
        _q73_rows.append({
            "Code": _c,
            "Polarity Role": _role,
            "Window Tagged Frames": _st["all_f"],
            "Raw Tag == GT (%)": round(_st["raw_hit"] / _st["all_f"] * 100, 1),
            "Implied Possessor == GT (All Frames %)": round(_imp_all, 1),
            "Implied Possessor == GT (Alive Frames %)": round(_imp_alive, 1),
            "Keep Feature? (≥60%)": "YES" if _imp_all >= 60.0 else "DROP (<60%)",
        })
    q73_df = pl.DataFrame(_q73_rows)

    # Q7.4: Team == "N/A" audit [FULL]
    q74_df = (
        tagging_df.group_by(["code", "type"])
        .agg([
            pl.len().alias("total_instances"),
            (pl.col("team") == "N/A").sum().alias("na_instances"),
            (pl.col("end_frame") - pl.col("start_frame")).sum().alias("total_frames"),
            pl.when(pl.col("team") == "N/A").then(pl.col("end_frame") - pl.col("start_frame")).otherwise(0).sum().alias("na_frames"),
        ])
        .filter(pl.col("na_instances") > 0)
        .sort("na_instances", descending=True)
    )

    mo.vstack([
        mo.md("## §7. Smart Tagging Priors `[FULL]` & `[GT-5MIN]`"),
        mo.md("### Q7.1 — Smart Tagging coverage by code & concurrency `[FULL]`\n"
              "**Finding:** Smart Tagging covers **82.34% of graded frames** (and **58.47%** of graded frames have **2 or more simultaneous tags** active, typically an offensive phase paired with the opposing team's defensive phase)."),
        q71_codes_df,
        q71_conc_df,
        mo.md("### Q7.2 — Set-piece boundary alignment with dead-ball runs `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** Set-piece tags are fixed **3.5s–35s windows** around restarts: for example, Kick Off (`11,097..11,222`) starts 88 frames before `11,185` and ends 37 frames after; Free Kick (`17,533..18,408`) spans across two separate `D` runs (`17,380..17,838` and `18,027..18,389`). Never copy a `SET PIECES` interval directly as `D`."),
        q72_df,
        mo.md("### Q7.3 — On-ball vs Off-ball implied possessor agreement `[GT-5MIN]`\n"
              "**Evidence Tag: `[GT-5MIN]`**\n"
              "**Finding:** Flipping the team polarity on defensive codes (`DEFENDING IN MIDDLE THIRD`, `DEFENSIVE TRANSITION`, `DEFENDING IN DEFENSIVE THIRD`) turns their agreement from **27.4% / 32.9% → 72.6% / 67.1%** (and **86.3% / 100.0%** on alive frames)!"),
        q73_df,
        mo.md("### Q7.4 — `Team = N/A` audit across all 638 instances `[FULL]`\n"
              "**Finding:** Exactly **36 instances** have `Team == 'N/A'`, all inside `SET PIECES`: **all 26 Throw-ins (100%)**, all 4 Kick Off (After Goal), all 2 Kick Off (Start), plus 4 of 12 Corner Kicks."),
        q74_df,
    ])
    return


@app.cell
def _(
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    N_FRAMES,
    active_away_per_frame,
    active_home_per_frame,
    active_players_per_frame,
    homography_df,
    mo,
    np,
    pl,
    px_norm,
    slot_is_away,
    slot_is_home,
):
    # =========================================================================
    # §8. COVERAGE AND GEOMETRY CHECKS [FULL] (Q8.1 - Q8.3)
    # =========================================================================
    _evidence = "FULL"
    _phases = [
        ("1. Pre-match (0–11,184)", 0, HALF1_START - 1),
        ("2. 1st Half Graded (11,185–90,651)", HALF1_START, HALF1_END),
        ("3. Halftime (90,652–110,807)", HALF1_END + 1, HALF2_START - 1),
        ("4. 2nd Half Graded (110,808–190,197)", HALF2_START, HALF2_END),
        ("5. Post-match (190,198–199,318)", HALF2_END + 1, N_FRAMES - 1),
    ]

    _reproj = homography_df["reprojection_error"].to_numpy()
    _q81_rows = []
    for _pname, _s, _e in _phases:
        _act = active_players_per_frame[_s : _e + 1]
        _act_h = active_home_per_frame[_s : _e + 1]
        _act_a = active_away_per_frame[_s : _e + 1]
        _rep = _reproj[_s : _e + 1]
        _q81_rows.append({
            "Match Phase": _pname,
            "Frames": _e - _s + 1,
            "Mean Tracked": round(float(_act.mean()), 2),
            "p10 / Med / p90 Tracked": f"{int(np.percentile(_act,10))} / {int(np.median(_act))} / {int(np.percentile(_act,90))}",
            "Mean Home / Away": f"{_act_h.mean():.1f} / {_act_a.mean():.1f}",
            "Frames < 10 Players (%)": round(float(np.mean(_act < 10) * 100), 2),
            "Frames > 22 Players (%)": round(float(np.mean(_act > 22) * 100), 2),
            "Homography Reproj Med / p95": f"{np.median(_rep):.3f} / {np.percentile(_rep,95):.3f}",
        })
    q81_df = pl.DataFrame(_q81_rows)

    # Q8.2: Does the coordinate frame flip at halftime? [FULL]
    _h1_hx = float(np.nanmean(px_norm[HALF1_START : HALF1_END + 1, slot_is_home]))
    _h1_ax = float(np.nanmean(px_norm[HALF1_START : HALF1_END + 1, slot_is_away]))
    _h2_hx = float(np.nanmean(px_norm[HALF2_START : HALF2_END + 1, slot_is_home]))
    _h2_ax = float(np.nanmean(px_norm[HALF2_START : HALF2_END + 1, slot_is_away]))

    q82_df = pl.DataFrame([
        {
            "Half": "1st Half (11,185–90,651)",
            "Barcelona (Home) Mean X": round(_h1_hx, 4),
            "Athletic (Away) Mean X": round(_h1_ax, 4),
            "Home - Away Mean ΔX": round(_h1_hx - _h1_ax, 4),
            "Home Attacking Direction": "Left → Right (+x)" if _h1_hx < _h1_ax else "Right → Left (-x)",
        },
        {
            "Half": "2nd Half (110,808–190,197)",
            "Barcelona (Home) Mean X": round(_h2_hx, 4),
            "Athletic (Away) Mean X": round(_h2_ax, 4),
            "Home - Away Mean ΔX": round(_h2_hx - _h2_ax, 4),
            "Home Attacking Direction": "Left → Right (+x)" if _h2_hx < _h2_ax else "Right → Left (-x)",
        },
    ])

    mo.vstack([
        mo.md(f"## §8. Coverage, Orientation, and Geometry Checks `[{_evidence}]`"),
        mo.md("### Q8.1 & Q8.3 — Player tracking density & homography error by match phase `[FULL]`"),
        q81_df,
        mo.md("### Q8.2 — Halftime coordinate orientation check `[FULL]`\n"
              "**Finding:** In the 1st Half, Barcelona (Home) is positioned to the left of Athletic (`mean_x = 0.4896` vs `0.5582`, `ΔX = -0.0686`, attacking Left → Right). "
              "In the 2nd Half, the ordering **reverses** (`mean_x = 0.5015` vs `0.4271`, `ΔX = +0.0744`, attacking Right → Left). "
              "**Decision:** Raw ATD coordinates are in fixed camera/pitch space; any feature that references own-half or attacking third **must flip `x → 1 - x` in the 2nd half**."),
        q82_df,
    ])
    return


if __name__ == "__main__":
    app.run()
