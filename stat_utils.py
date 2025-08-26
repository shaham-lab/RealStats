"""Statistical helpers re-exported from :mod:`utils`.

This thin module exists purely to provide a stable import path for the
statistical routines that live in ``utils.py``.
"""

from utils import (
    calculate_chi2,
    compute_chi2_matrix,
    calculate_chi2_and_corr,
    compute_chi2_and_corr_matrix,
    chi_square_independence_test,
    perform_ensemble_testing,
)

__all__ = [
    "calculate_chi2",
    "compute_chi2_matrix",
    "calculate_chi2_and_corr",
    "compute_chi2_and_corr_matrix",
    "chi_square_independence_test",
    "perform_ensemble_testing",
]
