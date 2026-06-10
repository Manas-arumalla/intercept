"""Base class for guidance laws.

A :class:`GuidanceLaw` is a stateful controller that conforms to the
:data:`intercept.core.entities.Controller` contract ``(t, own_state, world) -> control``. It
looks up the designated target's state from the ``world`` snapshot and delegates to
:meth:`command`. Subclasses implement the actual law (PN, APN, OGL, ...). Keeping a common base
means every law drops directly into an :class:`~intercept.core.Entity` and is benchmarked on
identical dynamics.

In P1 the target state passed to :meth:`command` is the *true* state (perfect information). From
P3 onward an estimator will sit between the sensor and the guidance law, and the same interface
will instead receive the *estimated* target state — no change to the laws themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Numerical floor for divisions by range / closing speed.
EPS = 1e-9


class GuidanceLaw(ABC):
    """Stateful guidance law bound to a named target.

    Parameters
    ----------
    target:
        Name of the target entity to home on (key into the ``world`` snapshot).
    """

    def __init__(self, target: str) -> None:
        self.target = target

    def reset(self) -> None:  # noqa: B027  (intentional optional hook, not abstract)
        """Reset any internal state (e.g., finite-difference memory). Override as needed."""

    def __call__(self, t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        target_state = np.asarray(world[self.target], dtype=float)
        return self.command(t, np.asarray(own_state, dtype=float), target_state)

    @abstractmethod
    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        """Return the commanded acceleration given own and target states."""

    # --- shared geometry helpers (planar; operate on [x, y, vx, vy] states) ---

    @staticmethod
    def _rel(own: Array, target: Array) -> tuple[Array, Array, float]:
        """Relative position ``r``, relative velocity ``v``, and range ``|r|``."""
        r = target[:2] - own[:2]
        v = target[2:4] - own[2:4]
        return r, v, float(np.linalg.norm(r))

    @staticmethod
    def _closing_speed(r: Array, v: Array, rng: float) -> float:
        if rng < EPS:
            return 0.0
        return float(-(r @ v) / rng)

    @staticmethod
    def _los_rate(r: Array, v: Array, rng: float) -> float:
        if rng < EPS:
            return 0.0
        return float(r[0] * v[1] - r[1] * v[0]) / (rng * rng)

    @staticmethod
    def _los_perp(r: Array, rng: float) -> Array:
        """Unit vector perpendicular to the LOS (LOS rotated +90 deg)."""
        if rng < EPS:
            return np.zeros(2)
        u = r / rng
        return np.array([-u[1], u[0]])
