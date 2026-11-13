"""Label-free full-match structural diagnostics for possession segmentation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from src.data.loader import (
    FPS,
    HALF1_END,
    HALF1_START,
    HALF2_END,
    HALF2_START,
    PUBLIC_WIN_END,
    PUBLIC_WIN_START,
)
from src.metrics import possession_metrics as pm

# Half-open [start, stop) intervals so halves are never bridged across halftime
STANDARD_SLICES: Dict[str, List[Tuple[int, int]]] = {
    "Public 5-min Window": [(PUBLIC_WIN_START, PUBLIC_WIN_END + 1)],
    "Rest of 1st Half": [(PUBLIC_WIN_END + 1, HALF1_END + 1)],
    "1st Half Graded": [(HALF1_START, HALF1_END + 1)],
    "2nd Half Graded": [(HALF2_START, HALF2_END + 1)],
    "Full Graded Match": [(HALF1_START, HALF1_END + 1), (HALF2_START, HALF2_END + 1)],
}


def _to_codes(labels_or_codes: np.ndarray) -> np.ndarray:
    """Convert string/object labels to int8 metric codes if needed."""
    arr = np.asarray(labels_or_codes)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int8, copy=False)
    return pm.label_codes(arr)


def diagnose_slice(
    codes: np.ndarray,
    runs: List[Tuple[int, int]],
    slice_name: str = "",
) -> Dict:
    """Compute structural segmentation and balance diagnostics on one slice."""
    codes = _to_codes(codes)
    raw_window = np.concatenate([codes[start:stop] for start, stop in runs])
    n_frames = len(raw_window)
    x_frame_pct = round(float(np.mean(raw_window == pm.X) * 100.0), 2) if n_frames else 0.0

    filled = pm.fill_abstentions(codes, runs)
    filled_window = np.concatenate([filled[start:stop] for start, stop in runs])

    h_count = int(np.sum(filled_window == pm.H))
    a_count = int(np.sum(filled_window == pm.A))
    d_count = int(np.sum(filled_window == pm.D))
    alive_count = h_count + a_count

    segs = pm.segments(filled, runs)
    spells = pm.possession_spells(filled, runs)
    spell_lengths = np.array([stop - start for start, stop, _ in spells], dtype=np.float64)
    d_lengths = np.array([stop - start for start, stop, code in segs if code == pm.D], dtype=np.float64)

    n_blocks_5min = n_frames / (FPS * 300.0) if n_frames else 0.0
    n_spells = len(spells)
    n_d_runs = len(d_lengths)

    return {
        "slice": slice_name,
        "frames": n_frames,
        "minutes": round(n_frames / (FPS * 60.0), 2),
        "n_spells": n_spells,
        "spells_per_5min": round(n_spells / n_blocks_5min, 2) if n_blocks_5min else 0.0,
        "mean_spell_s": round(float(np.mean(spell_lengths) / FPS), 2) if n_spells else 0.0,
        "median_spell_s": round(float(np.median(spell_lengths) / FPS), 2) if n_spells else 0.0,
        "p10_spell_s": round(float(np.percentile(spell_lengths, 10) / FPS), 2) if n_spells else 0.0,
        "p90_spell_s": round(float(np.percentile(spell_lengths, 90) / FPS), 2) if n_spells else 0.0,
        "short_spells_lt_1s_pct": round(float(np.mean(spell_lengths < FPS) * 100.0), 2) if n_spells else 0.0,
        "n_d_runs": n_d_runs,
        "d_runs_per_5min": round(n_d_runs / n_blocks_5min, 2) if n_blocks_5min else 0.0,
        "mean_d_s": round(float(np.mean(d_lengths) / FPS), 2) if n_d_runs else 0.0,
        "median_d_s": round(float(np.median(d_lengths) / FPS), 2) if n_d_runs else 0.0,
        "h_frame_pct": round(100.0 * h_count / n_frames, 2) if n_frames else 0.0,
        "a_frame_pct": round(100.0 * a_count / n_frames, 2) if n_frames else 0.0,
        "d_frame_pct": round(100.0 * d_count / n_frames, 2) if n_frames else 0.0,
        "x_frame_pct": x_frame_pct,
        "h_alive_share_pct": round(100.0 * h_count / alive_count, 2) if alive_count else 0.0,
    }


def diagnose_prediction(
    labels_or_codes: np.ndarray,
    model_name: str = "model",
) -> pl.DataFrame:
    """Evaluate diagnose_slice across all 5 standard match slices."""
    codes = _to_codes(labels_or_codes)
    rows = []
    for slice_name, runs in STANDARD_SLICES.items():
        row = {"model": model_name}
        row.update(diagnose_slice(codes, runs, slice_name=slice_name))
        rows.append(row)
    return pl.DataFrame(rows)


def _resolve_gt_codes(gt_csv_path: Optional[Union[str, Path]]) -> Optional[np.ndarray]:
    """Load public ground-truth codes if explicitly given or present at default path."""
    if gt_csv_path is None:
        default_gt = (
            Path(__file__).resolve().parent.parent.parent
            / "reference"
            / "evaluation"
            / "gt"
            / "BAR-ATH_possession_gt_public.csv"
        )
        if default_gt.exists():
            gt_csv_path = default_gt
    if gt_csv_path is not None and Path(gt_csv_path).exists():
        return pm.label_codes(pm.read_label_csv(str(gt_csv_path)))
    return None


def compare_models_full_match(
    models: Dict[str, np.ndarray],
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compare models across full-match structural slices and public 5-min GT metrics."""
    gt_codes = _resolve_gt_codes(gt_csv_path)
    detail_frames: List[pl.DataFrame] = []
    summary_rows: List[Dict] = []

    if gt_codes is not None:
        gt_row = {"model": "GT (Public 5-min)"}
        gt_row.update(diagnose_slice(gt_codes, STANDARD_SLICES["Public 5-min Window"], slice_name="Public 5-min Window"))
        detail_frames.append(pl.DataFrame([gt_row]))

    for model_name, preds in models.items():
        codes = _to_codes(preds)
        diag_df = diagnose_prediction(codes, model_name=model_name)
        detail_frames.append(diag_df)

        by_slice = {row["slice"]: row for row in diag_df.to_dicts()}
        full_r = by_slice["Full Graded Match"]
        h1_r = by_slice["1st Half Graded"]
        h2_r = by_slice["2nd Half Graded"]
        win_r = by_slice["Public 5-min Window"]

        summary = {
            "model": model_name,
            "full_spells_per_5min": full_r["spells_per_5min"],
            "full_mean_spell_s": full_r["mean_spell_s"],
            "full_median_spell_s": full_r["median_spell_s"],
            "full_short_lt_1s_pct": full_r["short_spells_lt_1s_pct"],
            "h1_spells_per_5min": h1_r["spells_per_5min"],
            "h2_spells_per_5min": h2_r["spells_per_5min"],
            "win_spells_per_5min": win_r["spells_per_5min"],
            "full_d_pct": full_r["d_frame_pct"],
            "win_d_pct": win_r["d_frame_pct"],
            "full_d_runs_per_5min": full_r["d_runs_per_5min"],
            "full_mean_d_s": full_r["mean_d_s"],
            "h1_h_alive_pct": h1_r["h_alive_share_pct"],
            "h2_h_alive_pct": h2_r["h_alive_share_pct"],
        }

        if gt_codes is not None:
            ev = pm.evaluate(gt_codes, codes)
            summary.update({
                "win_spellF1": round(ev["spell_f1_mean"], 4),
                "win_macroF1": round(ev["macro_f1"], 4),
                "win_alive_dead_f1": round(ev["alive_dead_f1"], 4),
                "win_team_acc": round(ev["team_acc_both_alive"], 4),
            })

        summary_rows.append(summary)

    detailed_df = pl.concat(detail_frames, how="vertical") if detail_frames else pl.DataFrame()
    summary_df = pl.DataFrame(summary_rows)
    return summary_df, detailed_df


if __name__ == "__main__":
    from src.data.loader import load_match_data
    from src.pipelines.baseline import run_baseline_pipeline

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(240)

    data = load_match_data()
    sweep_windows = (0, 25, 50, 75, 100, 125, 150)
    models = {}
    for w in sweep_windows:
        name = "baseline_raw" if w == 0 else f"baseline_smooth{w}"
        models[name] = run_baseline_pipeline(data=data, window=w).labels

    summary_df, detailed_df = compare_models_full_match(models)
    print("=== FULL-MATCH VS WINDOW SUMMARY (TARGET: ~16-19 spells/5min, 12-15s mean) ===")
    print(summary_df)

    official_names = {
        "GT (Public 5-min)",
        "baseline_raw",
        "baseline_smooth25",
        "baseline_smooth50",
        "baseline_smooth125",
    }
    print("\n=== PER-SLICE BREAKDOWN (GT + OFFICIAL BASELINES) ===")
    print(detailed_df.filter(pl.col("model").is_in(list(official_names))))

