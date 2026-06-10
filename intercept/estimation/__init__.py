"""State estimation, tracking, and trajectory prediction.

``ekf`` (Joseph-form EKF), ``ukf`` (sigma-point UKF), and ``imm`` (CV/CA/CT model bank) share a
common ``predict()`` / ``update(measurement) -> (x_hat, P)`` interface, with platform-navigation
error modeling for moving-seeker geometries.
"""

from intercept.estimation.base import Estimator
from intercept.estimation.imm import IMM, make_cv_ca_imm
from intercept.estimation.ins import INSError
from intercept.estimation.kalman import EKF, UKF
from intercept.estimation.models import STATE_DIM, nca_model, ncv_model

__all__ = [
    "Estimator",
    "EKF",
    "UKF",
    "IMM",
    "make_cv_ca_imm",
    "INSError",
    "ncv_model",
    "nca_model",
    "STATE_DIM",
]
