"""
Pantara — Physics-Aware Neural-network Tracking And Regression Analysis.

Matching-pursuit symbolic regression for physical laws.
"""

from pantara.core import (
    # Pipeline entry point
    pipeline,
    # Neural oracle
    OracleClassifier,
    # Constants
    N_CLASSES, DEVICE, FAMILIES, N_POINTS, MIN_SCORE,
    # Feature detection
    best_per_family_multispace, make_state_stats, make_oracle_input,
    # Block fitting
    fit_trig, detect_and_fit_power_law,
    fit_block, bloc_score, power_law_bloc_score,
    # Space transforms
    transform_space, inverse_transform_y,
    # Utilities
    corr_np, standardize,
)

import os as _os

MODEL_PATH = _os.path.join(_os.path.dirname(__file__), 'model.pt')

__version__ = '0.7.4'
__all__ = [
    'pipeline', 'OracleClassifier', 'MODEL_PATH',
    'N_CLASSES', 'DEVICE', 'FAMILIES', 'N_POINTS', 'MIN_SCORE',
    'best_per_family_multispace', 'make_state_stats', 'make_oracle_input',
    'fit_trig', 'detect_and_fit_power_law', 'fit_block',
    'bloc_score', 'power_law_bloc_score',
    'transform_space', 'inverse_transform_y',
    'corr_np', 'standardize',
]
