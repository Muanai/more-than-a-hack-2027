"""Learned weakly-supervised emission models and sequence decoders."""

from src.models.hsmm_decoder import decode_hsmm_3state, decode_viterbi_3state
from src.models.weak_emissions import (
    LearnedEmissions,
    compute_carried_margin,
    fit_predict_weak_emissions,
    prob_to_tanh_emission,
)

__all__ = [
    "LearnedEmissions",
    "compute_carried_margin",
    "decode_hsmm_3state",
    "decode_viterbi_3state",
    "fit_predict_weak_emissions",
    "prob_to_tanh_emission",
]

