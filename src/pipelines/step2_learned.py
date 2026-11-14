"""Step 2 Possession Pipelines: Weakly-Supervised Dropout Emissions + Viterbi / HSMM."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from src.data import MatchData, load_match_data
from src.evaluation import compare_models_full_match, diagnose_prediction
from src.features import PossessionFeatures, build_possession_features
from src.metrics import possession_metrics as pm
from src.models import (
    LearnedEmissions,
    decode_hsmm_3state,
    decode_viterbi_3state,
    fit_predict_weak_emissions,
    prob_to_tanh_emission,
)
from src.pipelines.ablation import run_ablation4_step1_physics
from src.pipelines.baseline import BaselineResult, save_submission


def _evaluate_and_wrap(
    labels: np.ndarray,
    model_name: str,
    window: int,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Evaluate predicted H/A/D/X labels on public GT and full-match slices."""
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

    metrics = None
    if gt_csv_path is not None and Path(gt_csv_path).exists():
        gt_df = pm.read_label_csv(str(gt_csv_path))
        metrics = pm.evaluate(pm.label_codes(gt_df), pm.label_codes(labels))

    raw_codes = pm.label_codes(labels).astype(np.int32)
    diagnostics = diagnose_prediction(labels, model_name=model_name)
    return BaselineResult(
        labels=labels,
        raw_codes=raw_codes,
        metrics=metrics,
        diagnostics=diagnostics,
        window=window,
    )


def run_ablation5_learned_viterbi(
    data: Optional[MatchData] = None,
    pf: Optional[PossessionFeatures] = None,
    emissions: Optional[LearnedEmissions] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Decode ball-dropout rich learned H/A emissions with calibrated 3-state Viterbi."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)
    if emissions is None:
        emissions = fit_predict_weak_emissions(data=data, pf=pf)

    emit_ha = prob_to_tanh_emission(emissions.p_ha_rich, temperature=3.5, smooth_width=15)
    labels = decode_viterbi_3state(
        n_frames=len(data.frames),
        emit_ha=emit_ha,
        emit_d=emissions.emit_d_morph,
        pen_ha=-22.0,
        pen_d=-24.0,
    )
    return _evaluate_and_wrap(labels, model_name="ablation5_learned_viterbi", window=15, gt_csv_path=gt_csv_path)


def run_ablation6_learned_hsmm(
    data: Optional[MatchData] = None,
    pf: Optional[PossessionFeatures] = None,
    emissions: Optional[LearnedEmissions] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Decode 2-expert learned H/A emissions with explicit-duration Block-5 HSMM."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)
    if emissions is None:
        emissions = fit_predict_weak_emissions(data=data, pf=pf)

    emit_ha = prob_to_tanh_emission(emissions.p_ha_mix, temperature=2.5, smooth_width=11)
    labels = decode_hsmm_3state(
        n_frames=len(data.frames),
        emit_ha=emit_ha,
        emit_d=emissions.emit_d_morph,
        pen_ha=-22.0,
        pen_d=-26.0,
        min_dur_ha_frames=25,
        min_dur_d_frames=45,
        dur_weight=0.7,
    )
    return _evaluate_and_wrap(labels, model_name="ablation6_learned_hsmm", window=11, gt_csv_path=gt_csv_path)


def run_step2_comparison(
    data: Optional[MatchData] = None,
    save_outputs: bool = True,
) -> Dict[str, BaselineResult]:
    """Fit Step 2 once under zero-leakage and compare Step 1 vs. Step 2 pipelines."""
    if data is None:
        data = load_match_data()
    pf = build_possession_features(data)
    emissions = fit_predict_weak_emissions(data=data, pf=pf)

    step1 = run_ablation4_step1_physics(data=data)
    step2_vit = run_ablation5_learned_viterbi(data=data, pf=pf, emissions=emissions)
    step2_hsmm = run_ablation6_learned_hsmm(data=data, pf=pf, emissions=emissions)

    results = {
        "ablation4_step1_physics": step1,
        "ablation5_learned_viterbi": step2_vit,
        "ablation6_learned_hsmm": step2_hsmm,
    }

    if save_outputs:
        pred_dir = Path("data") / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_submission(
            step2_vit.labels,
            out_path=pred_dir / "step2_learned_viterbi.csv",
            ungraded_to_x=True,
            graded_mask=data.graded_mask,
        )
        save_submission(
            step2_hsmm.labels,
            out_path=pred_dir / "step2_learned_hsmm.csv",
            ungraded_to_x=True,
            graded_mask=data.graded_mask,
        )

    print(
        f"Zero-leakage H/A weak anchors: n={emissions.n_ha_anchors:,} "
        f"(H share={100.0 * emissions.ha_anchor_home_share:.1f}%)"
    )
    print("\n--- Public 5-min GT Window Comparison ---")
    for name, res in results.items():
        if res.metrics is not None:
            m = res.metrics
            print(
                f"{name:26s} | spellF1={m['spell_f1_mean']:.4f} "
                f"macroF1={m['macro_f1']:.4f} AD_F1={m['alive_dead_f1']:.4f} "
                f"teamAcc={m['team_acc_both_alive']:.4f} spells={m['n_pred_spells']}/{m['n_gt_spells']} "
                f"meanDur={m['mean_spell_duration_s']:.1f}s shareErr={m['share_err']:.4f}"
            )

    comp = compare_models_full_match({k: v.labels for k, v in results.items()})
    print("\n--- Full-Match Structural Diagnostics ---")
    print(comp)
    return results


if __name__ == "__main__":
    run_step2_comparison(save_outputs=True)

