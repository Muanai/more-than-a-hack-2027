"""Step 3 Possession Pipelines: Outer-EM Self-Training & Short-Stoppage Recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from src.data import MatchData, load_match_data
from src.evaluation import compare_models_full_match
from src.features import PossessionFeatures, build_possession_features
from src.models import (
    Step3Emissions,
    decode_hsmm_3state,
    decode_viterbi_3state,
    fit_predict_step3_em,
    fit_predict_weak_emissions,
    prob_to_tanh_emission,
)
from src.pipelines.ablation import run_ablation4_step1_physics
from src.pipelines.baseline import BaselineResult, save_submission
from src.pipelines.step2_learned import (
    _evaluate_and_wrap,
    run_ablation5_learned_viterbi,
    run_ablation6_learned_hsmm,
)


def run_ablation7_step3_viterbi(
    data: Optional[MatchData] = None,
    pf: Optional[PossessionFeatures] = None,
    emissions: Optional[Step3Emissions] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Decode Step 3 Outer-EM 2-expert emissions + enhanced D with calibrated Viterbi."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)
    if emissions is None:
        emissions = fit_predict_step3_em(data=data, pf=pf)

    emit_ha = prob_to_tanh_emission(emissions.p_ha_em_mix, temperature=3.0, smooth_width=15)
    labels = decode_viterbi_3state(
        n_frames=len(data.frames),
        emit_ha=emit_ha,
        emit_d=emissions.emit_d_enh,
        pen_ha=-24.0,
        pen_d=-24.0,
    )
    return _evaluate_and_wrap(labels, model_name="ablation7_step3_em_viterbi", window=15, gt_csv_path=gt_csv_path)


def run_ablation8_step3_hsmm(
    data: Optional[MatchData] = None,
    pf: Optional[PossessionFeatures] = None,
    emissions: Optional[Step3Emissions] = None,
    gt_csv_path: Optional[Union[str, Path]] = None,
) -> BaselineResult:
    """Decode Step 3 Outer-EM 2-expert emissions + enhanced D with Block-5 HSMM."""
    if data is None:
        data = load_match_data()
    if pf is None:
        pf = build_possession_features(data)
    if emissions is None:
        emissions = fit_predict_step3_em(data=data, pf=pf)

    emit_ha = prob_to_tanh_emission(emissions.p_ha_em_mix, temperature=3.0, smooth_width=15)
    labels = decode_hsmm_3state(
        n_frames=len(data.frames),
        emit_ha=emit_ha,
        emit_d=emissions.emit_d_enh,
        pen_ha=-26.0,
        pen_d=-26.0,
        min_dur_ha_frames=25,
        min_dur_d_frames=45,
        dur_weight=0.7,
    )
    return _evaluate_and_wrap(labels, model_name="ablation8_step3_em_hsmm", window=15, gt_csv_path=gt_csv_path)


def run_step3_comparison(
    data: Optional[MatchData] = None,
    save_outputs: bool = True,
) -> Dict[str, BaselineResult]:
    """Compare Step 1, Step 2, and Step 3 pipelines under strict zero-leakage."""
    if data is None:
        data = load_match_data()
    pf = build_possession_features(data)

    em_s2 = fit_predict_weak_emissions(data=data, pf=pf)
    em_s3 = fit_predict_step3_em(data=data, pf=pf)

    results = {
        "ablation4_step1_physics": run_ablation4_step1_physics(data=data),
        "ablation5_learned_viterbi": run_ablation5_learned_viterbi(data=data, pf=pf, emissions=em_s2),
        "ablation6_learned_hsmm": run_ablation6_learned_hsmm(data=data, pf=pf, emissions=em_s2),
        "ablation7_step3_em_viterbi": run_ablation7_step3_viterbi(data=data, pf=pf, emissions=em_s3),
        "ablation8_step3_em_hsmm": run_ablation8_step3_hsmm(data=data, pf=pf, emissions=em_s3),
    }

    if save_outputs:
        pred_dir = Path("data") / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_submission(
            results["ablation7_step3_em_viterbi"].labels,
            out_path=pred_dir / "step3_em_viterbi.csv",
            ungraded_to_x=True,
            graded_mask=data.graded_mask,
        )
        save_submission(
            results["ablation8_step3_em_hsmm"].labels,
            out_path=pred_dir / "step3_em_hsmm.csv",
            ungraded_to_x=True,
            graded_mask=data.graded_mask,
        )

    print(
        f"Step 3 Outer-EM zero-leakage coverage: R1={em_s3.n_r1_anchors:,} -> "
        f"EM={em_s3.n_em_frames:,} non-window frames | short D cores={em_s3.n_short_stop_cores}"
    )
    print("\n--- Public 5-min GT Window Comparison (Step 1 -> Step 2 -> Step 3) ---")
    for name, res in results.items():
        if res.metrics is not None:
            m = res.metrics
            print(
                f"{name:27s} | spellF1={m['spell_f1_mean']:.4f} "
                f"macroF1={m['macro_f1']:.4f} AD_F1={m['alive_dead_f1']:.4f} "
                f"teamAcc={m['team_acc_both_alive']:.4f} spells={m['n_pred_spells']}/{m['n_gt_spells']} "
                f"meanDur={m['mean_spell_duration_s']:.1f}s shareErr={m['share_err']:.4f}"
            )

    summary, _ = compare_models_full_match({k: v.labels for k, v in results.items()})
    print("\n--- Full-Match Structural Summary ---")
    for row in summary.to_dicts():
        print(
            f"{row['model']:27s} | full sp/5m={row['full_spells_per_5min']:5.2f} "
            f"mean={row['full_mean_spell_s']:5.2f}s <1s={row['full_short_lt_1s_pct']:4.1f}% "
            f"H1/H2={row['h1_spells_per_5min']:5.2f}/{row['h2_spells_per_5min']:5.2f} "
            f"fullD={row['full_d_pct']:5.2f}% winD={row['win_d_pct']:5.2f}%"
        )
    return results


if __name__ == "__main__":
    run_step3_comparison(save_outputs=True)

