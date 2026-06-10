"""Discrete-time motion models on a ``[pos(n), vel(n), acc(n)]`` target state (n = 2 or 3).

Two models, sharing the state dimension so they can be mixed in an IMM (see :mod:`.imm`):

* **NCV** (nearly-constant-velocity): position integrates velocity; acceleration is *not* fed into
  velocity and is driven only by small process noise — it stays near zero. The quiescent model.
* **NCA** (nearly-constant-acceleration): full constant-acceleration kinematics (acceleration feeds
  velocity feeds position) with larger process noise — the maneuver model.

Each builder returns ``(F, Q)`` for a given step ``dt`` and continuous process-noise spectral
density ``q`` (per axis), using the standard white-noise-jerk / white-noise-acceleration forms. The
axis count ``ndim`` defaults to 2 (state ``[x, y, vx, vy, ax, ay]``); ``ndim=3`` gives the 3-D state
``[x, y, z, vx, vy, vz, ax, ay, az]`` for 3-D tracking (the layout the 3-D plants and sensors use).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

STATE_DIM = 6  # default 2-D state [x, y, vx, vy, ax, ay]


def ncv_model(dt: float, q: float = 1.0, ndim: int = 2) -> tuple[Array, Array]:
    """Nearly-constant-velocity model (acceleration states present but uncoupled, low noise)."""
    n = 3 * ndim
    F = np.eye(n)
    Q = np.zeros((n, n))
    for i in range(ndim):
        p, v, a = i, ndim + i, 2 * ndim + i
        F[p, v] = dt
        # acceleration is NOT propagated into velocity/position (constant-velocity assumption);
        # white-noise-acceleration process noise on the (pos, vel) channel.
        Q[p, p] = q * dt**3 / 3.0
        Q[p, v] = Q[v, p] = q * dt**2 / 2.0
        Q[v, v] = q * dt
        Q[a, a] = 1e-6  # tiny noise to keep the acceleration covariance non-singular
    return F, Q


def nca_model(dt: float, q: float = 50.0, ndim: int = 2) -> tuple[Array, Array]:
    """Nearly-constant-acceleration model (full CA kinematics, white-noise-jerk process noise)."""
    n = 3 * ndim
    F = np.eye(n)
    Q = np.zeros((n, n))
    for i in range(ndim):
        p, v, a = i, ndim + i, 2 * ndim + i
        F[p, v] = dt
        F[p, a] = 0.5 * dt**2
        F[v, a] = dt
        # white-noise-jerk discrete process noise per axis on (pos, vel, acc)
        Q[p, p] = q * dt**5 / 20.0
        Q[p, v] = Q[v, p] = q * dt**4 / 8.0
        Q[p, a] = Q[a, p] = q * dt**3 / 6.0
        Q[v, v] = q * dt**3 / 3.0
        Q[v, a] = Q[a, v] = q * dt**2 / 2.0
        Q[a, a] = q * dt
    return F, Q
