"""Tests for IMM-mode-adaptive guidance (estimator-driven law blending)."""

from __future__ import annotations

import numpy as np

from intercept.adversary import step_maneuver
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.estimation import make_cv_ca_imm
from intercept.guidance import AugmentedPN, ModeAdaptiveGuidance, true_pn
from intercept.sensors import Radar


def _engage(factory, t_break: float = 4.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    idyn = AeroMissile2D(a_max=40 * G0, tau=0.2)
    edyn = AeroMissile2D(a_max=25 * G0, tau=0.3)
    intc = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], [1000.0, 80.0]),
        controller=factory(rng),
        role="interceptor",
    )
    tgt = Entity(
        "target",
        edyn,
        edyn.initial_state([8000.0, 700.0], [-700.0, 0.0]),
        controller=step_maneuver(accel=20 * G0, t_start=t_break),
        role="target",
    )
    return Engagement(
        [intc, tgt],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=16.0,
        kill_radius=20.0,
    ).run()


def _adaptive(rng):
    return ModeAdaptiveGuidance(
        "target",
        Radar(sigma_range=15.0, sigma_bearing=0.004),
        lambda x0, P0: make_cv_ca_imm(x0, P0, p_stay=0.995),  # sticky: integrate mode evidence
        true_pn("target", N=4.0),
        AugmentedPN("target", N=4.0),
        rng,
    )


def test_mode_belief_rises_after_the_break():
    """The logged maneuver-mode probability is low pre-break and high after the target breaks."""
    guid_holder = {}

    def factory(rng):
        guid_holder["g"] = _adaptive(rng)
        return guid_holder["g"]

    # Break early so both modes are observed. (This early-sustained-break case is the documented
    # envelope limit — detection lag can cost the intercept — so only the *belief* is asserted.)
    _engage(factory, t_break=2.0)
    mu = np.array(guid_holder["g"].mu_history)  # (t, P(maneuver))
    pre = mu[(mu[:, 0] > 0.8) & (mu[:, 0] < 1.8), 1]  # settled cruise window
    post = mu[mu[:, 0] > 2.8, 1]  # after the 2 s break (+detection lag)
    assert pre.size and post.size
    assert pre.mean() < 0.5 < post.mean()  # belief actually switches


def test_adaptive_matches_apn_intercept_at_lower_effort():
    """Same intercept as always-on APN on cruise-then-break, with less control effort."""

    def apn(rng):
        from intercept.estimation import EKF, nca_model
        from intercept.guidance import EstimatingGuidance

        return EstimatingGuidance(
            "target",
            Radar(sigma_range=15.0, sigma_bearing=0.004),
            lambda x0, P0: EKF(lambda d: nca_model(d, 50.0), x0, P0),
            AugmentedPN("target", N=4.0),
            rng,
        )

    eff_adaptive, eff_apn = [], []
    for s in range(4):
        ra = _engage(_adaptive, seed=s)
        rb = _engage(apn, seed=s)
        assert ra.intercepted and rb.intercepted
        eff_adaptive.append(ra.control_effort("interceptor"))
        eff_apn.append(rb.control_effort("interceptor"))
    assert np.mean(eff_adaptive) < np.mean(eff_apn)
