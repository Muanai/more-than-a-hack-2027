"""Learned weakly-supervised emission models and sequence decoders."""

from src.models.em_refinement import (
    Step3Emissions,
    build_enhanced_d_emission,
    build_step3_feature_matrix,
    fit_predict_step3_em,
)
from src.models.hsmm_decoder import decode_hsmm_3state, decode_viterbi_3state
from src.models.weak_emissions import (
    LearnedEmissions,
    compute_carried_margin,
    fit_predict_weak_emissions,
    prob_to_tanh_emission,
)

__all__ = [
    "LearnedEmissions",
    "Step3Emissions",
    "build_enhanced_d_emission",
    "build_step3_feature_matrix",
    "compute_carried_margin",
    "decode_hsmm_3state",
    "decode_viterbi_3state",
    "fit_predict_step3_em",
    "fit_predict_weak_emissions",
    "prob_to_tanh_emission",
]
