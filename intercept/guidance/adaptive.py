"""Mode-adaptive guidance: the IMM's maneuver belief arbitrates between guidance laws.

A novel estimation-aware composition: an interceptor-mounted sensor feeds an **IMM** (CV + CA model
bank), and each step the commanded acceleration is the **mode-probability-weighted blend** of a
*quiescent* law (efficient, e.g. True PN) and a *maneuver* law (aggressive, e.g. Augmented PN or
sliding-mode):

    a  =  μ_CV · a_quiescent(x̂)  +  μ_CA · a_maneuver(x̂)

where ``μ`` are the IMM mode probabilities and ``x̂`` the fused estimate. Against a target that
cruises and then breaks, this flies cheap PN while the target is quiet and hardens automatically the
moment the filter *detects* the maneuver — robustness when needed, efficiency when not. The blend is
smooth (no chattering switches), and the arbitration signal is exactly the statistic the IMM already
computes; no extra tuning beyond the filter itself.

Sensing follows :class:`~intercept.guidance.estimating.EstimatingGuidance` (seeker on the
interceptor, RNG injected per trial, reproducible); ``mu_history`` is logged for analysis/plots.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller
from intercept.estimation.imm import IMM
from intercept.sensors.base import Sensor

Array = NDArray[np.float64]


class ModeAdaptiveGuidance:
    """Blend two guidance laws by the IMM's quiescent/maneuver mode probabilities.

    Parameters
    ----------
    target:
        Name of the target entity in the world snapshot.
    sensor:
        Measurement model (e.g. :class:`~intercept.sensors.radar.Radar`).
    imm_factory:
        ``factory(x0, P0) -> IMM`` whose **model 0 is quiescent (CV)** and **model 1 is maneuver
        (CA)** — e.g. :func:`~intercept.estimation.imm.make_cv_ca_imm`.
    quiescent_law, maneuver_law:
        Controllers bound to ``target`` (e.g. ``true_pn`` and ``AugmentedPN``/``sliding_mode``).
    rng:
        Generator for sensor noise (fixed per trial; reproducible).
    """

    def __init__(
        self,
        target: str,
        sensor: Sensor,
        imm_factory: Callable[[Array, Array], IMM],
        quiescent_law: Controller,
        maneuver_law: Controller,
        rng: np.random.Generator,
        *,
        sharpness: float = 3.0,
        init_pos_std: float = 50.0,
        init_vel_std: float = 300.0,
        init_acc_std: float = 100.0,
    ) -> None:
        self.target = target
        self.sensor = sensor
        self.imm_factory = imm_factory
        self.quiescent_law = quiescent_law
        self.maneuver_law = maneuver_law
        self.rng = rng
        # Arbitration sharpness γ: blend weight w = μ^γ / (μ^γ + (1−μ)^γ). γ=1 is the linear blend;
        # γ>1 commits harder once the belief tilts (more PN when quiet, more maneuver-law when not),
        # countering detection-lag dilution without a hard (chattering) switch.
        self.sharpness = float(sharpness)
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std
        self.init_acc_std = init_acc_std
        self.imm: IMM | None = None
        self._last_t: float | None = None
        self.mu_history: list[tuple[float, float]] = []  # (t, P(maneuver mode))

    @property
    def _ndim(self) -> int:
        return int(getattr(self.sensor, "pos_dim", 2))

    def _initialize(self, sensor_pos: Array, z: Array) -> None:
        n = self._ndim
        invert = getattr(self.sensor, "invert", None)
        pos0 = (
            np.asarray(invert(sensor_pos, z), dtype=float)[:n]
            if invert is not None
            else np.asarray(sensor_pos, dtype=float)[:n].copy()
        )
        x0 = np.concatenate([pos0, np.zeros(2 * n)])
        diag = [self.init_pos_std**2] * n + [self.init_vel_std**2] * n + [self.init_acc_std**2] * n
        self.imm = self.imm_factory(x0, np.diag(diag))

    def __call__(self, t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        n = self._ndim
        sensor_pos = np.asarray(own_state, dtype=float)[:n]
        z = self.sensor.measure(sensor_pos, np.asarray(world[self.target], float)[:n], self.rng)
        if self.imm is None:
            self._initialize(sensor_pos, z)
        else:
            dt = t - (self._last_t if self._last_t is not None else t)
            if dt > 0:
                self.imm.predict(dt)
            self.imm.update(z, self.sensor, sensor_pos)
        self._last_t = t

        assert self.imm is not None
        mu = np.asarray(self.imm.mode_probabilities, dtype=float)
        mu_maneuver = float(mu[1]) if mu.size > 1 else 0.0
        self.mu_history.append((t, mu_maneuver))

        est_world = dict(world)
        est_world[self.target] = self.imm.target_state()
        a_q = np.asarray(self.quiescent_law(t, own_state, est_world), dtype=float)
        a_m = np.asarray(self.maneuver_law(t, own_state, est_world), dtype=float)
        g = self.sharpness
        num = mu_maneuver**g
        w = num / (num + (1.0 - mu_maneuver) ** g)
        return (1.0 - w) * a_q + w * a_m
