"""Tests for sensors, EKF/UKF/IMM filters, and estimating guidance.

Covers measurement-model correctness, filter convergence on noisy data, statistical consistency
(NEES), IMM mode-probability behavior, and that closing the sense->estimate->guide loop still
intercepts (with noise-degraded but reasonable accuracy).
"""

import numpy as np
import pytest

from intercept.core import Engagement, Entity, PointMass2D
from intercept.estimation import EKF, UKF, make_cv_ca_imm, nca_model, ncv_model
from intercept.guidance import EstimatingGuidance, true_pn
from intercept.sensors import IRSeeker, Radar
from intercept.sensors.base import wrap_to_pi

# --- sensors ---------------------------------------------------------------


def test_radar_measurement_noise_free():
    radar = Radar(sigma_range=10.0, sigma_bearing=0.01)
    z = radar.h(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert z[0] == pytest.approx(5.0)
    assert z[1] == pytest.approx(np.arctan2(4.0, 3.0))


def test_radar_jacobian_matches_finite_difference():
    radar = Radar()
    sp = np.array([100.0, -50.0])
    tp = np.array([1200.0, 300.0])
    J = radar.jacobian(sp, tp)
    eps = 1e-4
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        num = radar.residual(radar.h(sp, tp + d), radar.h(sp, tp - d)) / (2 * eps)
        assert np.allclose(J[:, k], num, atol=1e-4)


def test_radar_noise_statistics():
    radar = Radar(sigma_range=15.0, sigma_bearing=0.02)
    rng = np.random.default_rng(0)
    sp, tp = np.array([0.0, 0.0]), np.array([2000.0, 0.0])
    zs = np.array([radar.measure(sp, tp, rng) for _ in range(5000)])
    truth = radar.h(sp, tp)
    assert np.allclose(zs.mean(axis=0), truth, atol=[1.0, 1e-3])
    assert zs[:, 0].std() == pytest.approx(15.0, rel=0.1)
    assert zs[:, 1].std() == pytest.approx(0.02, rel=0.1)


def test_ir_bearing_only():
    ir = IRSeeker(sigma_bearing=0.01)
    assert ir.dim == 1
    z = ir.h(np.array([0.0, 0.0]), np.array([0.0, 10.0]))
    assert z[0] == pytest.approx(np.pi / 2)


def test_wrap_to_pi():
    assert wrap_to_pi(2 * np.pi + 0.5) == pytest.approx(0.5)
    assert wrap_to_pi(-2 * np.pi - 0.5) == pytest.approx(-0.5)
    assert wrap_to_pi(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    # ±π are the same angle; wrapping a boundary lands on one of them.
    assert abs(wrap_to_pi(3 * np.pi)) == pytest.approx(np.pi)


# --- filters ---------------------------------------------------------------


def _track_constant_velocity(filter_cls):
    """Run a filter against a constant-velocity target observed by a noisy radar."""
    rng = np.random.default_rng(1)
    radar = Radar(sigma_range=10.0, sigma_bearing=0.005)
    sensor_pos = np.array([0.0, 0.0])
    true = np.array([2000.0, 500.0, -150.0, 30.0, 0.0, 0.0])  # x,y,vx,vy,ax,ay
    dt = 0.1

    x0 = np.array([2000.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([100.0, 100.0, 300.0**2, 300.0**2, 50.0**2, 50.0**2])
    est = filter_cls(lambda d: ncv_model(d, q=1.0), x0, P0)

    errs = []
    for k in range(120):
        true = true.copy()
        true[:2] = true[:2] + true[2:4] * dt
        z = radar.measure(sensor_pos, true[:2], rng)
        est.predict(dt)
        est.update(z, radar, sensor_pos)
        if k > 60:
            errs.append(np.linalg.norm(est.position - true[:2]))
    return float(np.mean(errs))


def test_ekf_converges_on_constant_velocity():
    assert _track_constant_velocity(EKF) < 30.0


def test_ukf_converges_on_constant_velocity():
    assert _track_constant_velocity(UKF) < 30.0


def test_ekf_velocity_is_estimated():
    rng = np.random.default_rng(2)
    radar = Radar(sigma_range=8.0, sigma_bearing=0.004)
    sp = np.array([0.0, 0.0])
    true = np.array([3000.0, 0.0, -200.0, 50.0, 0.0, 0.0])
    dt = 0.1
    est = EKF(
        lambda d: ncv_model(d, 1.0),
        np.array([3000.0, 0.0, 0, 0, 0, 0]),
        np.diag([100.0, 100.0, 300**2, 300**2, 1.0, 1.0]),
    )
    for _ in range(150):
        true[:2] += true[2:4] * dt
        z = radar.measure(sp, true[:2], rng)
        est.predict(dt)
        est.update(z, radar, sp)
    assert np.allclose(est.velocity, true[2:4], atol=25.0)


def test_ekf_nees_is_consistent():
    # Average NEES should be near the state dimension (6) for a well-tuned filter.
    rng = np.random.default_rng(3)
    radar = Radar(sigma_range=10.0, sigma_bearing=0.005)
    sp = np.array([0.0, 0.0])
    neess = []
    for _trial in range(20):
        true = np.array([2500.0, 200.0, -180.0, 20.0, 0.0, 0.0])
        est = EKF(
            lambda d: ncv_model(d, 1.0),
            np.array([2500.0, 200.0, 0, 0, 0, 0]),
            np.diag([100.0, 100.0, 250**2, 250**2, 10.0, 10.0]),
        )
        dt = 0.1
        for _ in range(100):
            true[:2] += true[2:4] * dt
            est.predict(dt)
            est.update(radar.measure(sp, true[:2], rng), radar, sp)
        neess.append(est.nees(true))
    # Loose bounds: consistent filters keep mean NEES within a few x the state dim.
    assert 1.0 < np.mean(neess) < 40.0


def test_imm_favors_ca_model_under_maneuver():
    rng = np.random.default_rng(4)
    radar = Radar(sigma_range=8.0, sigma_bearing=0.004)
    sp = np.array([0.0, 0.0])
    imm = make_cv_ca_imm(
        np.array([2000.0, 0.0, -150.0, 0.0, 0.0, 0.0]),
        np.diag([50.0, 50.0, 50**2, 50**2, 50**2, 50**2]),
    )
    true = np.array([2000.0, 0.0, -150.0, 0.0, 80.0, 60.0])  # constant acceleration (maneuver)
    dt = 0.1
    for _ in range(60):
        true[2:4] += true[4:6] * dt
        true[:2] += true[2:4] * dt
        imm.predict(dt)
        imm.update(radar.measure(sp, true[:2], rng), radar, sp)
    # Under a sustained maneuver, the CA model (index 1) should carry most of the probability.
    assert imm.mode_probabilities[1] > imm.mode_probabilities[0]


def test_imm_transition_validation():
    with pytest.raises(ValueError):
        from intercept.estimation import IMM

        IMM([EKF(lambda d: ncv_model(d), np.zeros(6), np.eye(6))], np.eye(2))


# --- estimating guidance ---------------------------------------------------


def test_estimating_guidance_still_intercepts_with_noise():
    rng = np.random.default_rng(5)
    radar = Radar(sigma_range=15.0, sigma_bearing=0.005)
    guidance = EstimatingGuidance(
        "target",
        radar,
        lambda x0, P0: EKF(lambda d: nca_model(d, 50.0), x0, P0),
        true_pn("target", N=4.0),
        rng,
    )
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=300.0),
        np.array([0.0, 0.0, 700.0, 0.0]),
        controller=guidance,
        role="interceptor",
    )
    target = Entity("target", PointMass2D(), np.array([4000.0, 600.0, -250.0, 0.0]), role="target")
    result = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=30.0,
        kill_radius=20.0,
    ).run()
    # With a noisy seeker the intercept is harder, but a good filter should still bring it home.
    assert result.miss_distance < 50.0


# --- 3-D estimation --------------------------------------------------------


def test_motion_models_3d_shapes_and_2d_unchanged():
    F2, Q2 = ncv_model(0.1, q=1.0)  # default ndim=2
    assert F2.shape == (6, 6) and Q2.shape == (6, 6)
    F3, Q3 = nca_model(0.1, q=50.0, ndim=3)
    assert F3.shape == (9, 9) and Q3.shape == (9, 9)
    # 3-D CA kinematics: position couples to velocity and acceleration on each axis.
    for p, v, a in ((0, 3, 6), (1, 4, 7), (2, 5, 8)):
        assert F3[p, v] == pytest.approx(0.1)
        assert F3[p, a] == pytest.approx(0.5 * 0.1**2)
        assert F3[v, a] == pytest.approx(0.1)


def test_radar3d_roundtrip_and_residual_wrap():
    from intercept.sensors import Radar3D

    radar = Radar3D()
    sp = np.zeros(3)
    tgt = np.array([3000.0, 1200.0, 2000.0])
    z = radar.h(sp, tgt)
    assert np.allclose(radar.invert(sp, z), tgt, atol=1e-6)  # h/invert are inverses
    assert radar.pos_dim == 3 and radar.dim == 3
    # azimuth residual wraps across ±π (no 2π jump)
    d = radar.residual(np.array([0.0, 3.13, 0.0]), np.array([0.0, -3.13, 0.0]))
    assert abs(d[1]) < 0.05


def test_ukf_tracks_3d_maneuvering_target():
    from intercept.adversary import barrel_roll
    from intercept.core import G0, AeroMissile3D
    from intercept.core.integrators import integrate_rk4
    from intercept.sensors import Radar3D

    rng = np.random.default_rng(1)
    plant = AeroMissile3D(a_max=20 * G0, tau=0.3)
    x = plant.initial_state([8000.0, 1500.0, 4000.0], [-700.0, 0.0, -30.0])
    ctrl = barrel_roll(accel=15 * G0, rate=1.3)
    radar = Radar3D(sigma_range=25.0, sigma_az=0.008, sigma_el=0.008)
    sp = np.zeros(3)
    dt = 0.05
    x0 = np.concatenate(
        [radar.invert(sp, radar.measure(sp, x[:3], rng)), [-700.0, 0.0, -30.0], [0.0, 0.0, 0.0]]
    )
    P0 = np.diag([200.0] * 6 + [100.0] * 3) ** 2
    ukf = UKF(lambda d: nca_model(d, q=200.0, ndim=3), x0, P0)
    errs, raw = [], []
    for k in range(120):
        x = integrate_rk4(plant, k * dt, x, ctrl(k * dt, x, {}), dt)
        ukf.predict(dt)
        z = radar.measure(sp, x[:3], rng)
        ukf.update(z, radar, sp)
        if k > 20:
            errs.append(np.linalg.norm(ukf.x[:3] - x[:3]))
            raw.append(np.linalg.norm(radar.invert(sp, z) - x[:3]))
    rmse = np.sqrt(np.mean(np.square(errs)))
    raw_rmse = np.sqrt(np.mean(np.square(raw)))
    assert rmse < raw_rmse  # the filter beats raw measurements
    assert rmse < 70.0  # tracks the 3-D maneuver to within ~tens of m


def test_imm_3d_tracks_and_raises_maneuver_probability():
    from intercept.adversary import barrel_roll
    from intercept.core import G0, AeroMissile3D
    from intercept.core.integrators import integrate_rk4
    from intercept.estimation import make_cv_ca_imm
    from intercept.sensors import Radar3D

    rng = np.random.default_rng(2)
    plant = AeroMissile3D(a_max=20 * G0, tau=0.3)
    x = plant.initial_state([8000.0, 1500.0, 4000.0], [-700.0, 0.0, -30.0])
    ctrl = barrel_roll(accel=15 * G0, rate=1.3)
    radar = Radar3D(sigma_range=25.0, sigma_az=0.008, sigma_el=0.008)
    sp = np.zeros(3)
    dt = 0.05
    x0 = np.concatenate(
        [radar.invert(sp, radar.measure(sp, x[:3], rng)), [-700.0, 0.0, -30.0], [0.0, 0.0, 0.0]]
    )
    P0 = np.diag([200.0] * 6 + [100.0] * 3) ** 2
    # A second pure-NCV EKF for comparison (the IMM should beat it on the maneuver).
    from intercept.estimation import EKF

    ncv = EKF(lambda d: ncv_model(d, q=5.0, ndim=3), x0.copy(), P0.copy())
    imm = make_cv_ca_imm(x0, P0, q_cv=5.0, q_ca=200.0, ndim=3)
    assert imm.x.shape == (9,) and imm.mode_probabilities.shape == (2,)
    e_imm, e_ncv, mu_ca = [], [], []
    for k in range(120):
        x = integrate_rk4(plant, k * dt, x, ctrl(k * dt, x, {}), dt)
        z = radar.measure(sp, x[:3], rng)
        imm.predict(dt)
        imm.update(z, radar, sp)
        ncv.predict(dt)
        ncv.update(z, radar, sp)
        mu_ca.append(imm.mode_probabilities[1])
        if k > 20:
            e_imm.append(np.linalg.norm(imm.x[:3] - x[:3]))
            e_ncv.append(np.linalg.norm(ncv.x[:3] - x[:3]))
    rmse_imm = np.sqrt(np.mean(np.square(e_imm)))
    assert rmse_imm < 70.0  # tracks the 3-D maneuver
    assert rmse_imm < np.sqrt(np.mean(np.square(e_ncv)))  # beats a pure NCV EKF
    assert max(mu_ca) > 0.4  # maneuver model meaningfully engaged


def test_estimating_guidance_3d_closes_the_loop():
    """sense -> estimate -> guide in 3-D: Radar3D + NCA-UKF feeding True PN-3D still intercepts."""
    from intercept.core import G0, AeroMissile3D, Engagement, Entity
    from intercept.guidance import EstimatingGuidance, true_pn_3d
    from intercept.sensors import Radar3D

    rng = np.random.default_rng(7)
    radar = Radar3D(sigma_range=20.0, sigma_az=0.005, sigma_el=0.005)
    guidance = EstimatingGuidance(
        "target",
        radar,
        lambda x0, P0: UKF(lambda d: nca_model(d, 150.0, ndim=3), x0, P0),
        true_pn_3d("target", N=4.0),
        rng,
    )
    intc = AeroMissile3D(a_max=45 * G0, tau=0.2)
    interceptor = Entity(
        "interceptor",
        intc,
        intc.initial_state([0, 0, 0], [1000.0, 0.0, 150.0]),
        controller=guidance,
        role="interceptor",
    )
    tgt = AeroMissile3D(a_max=20 * G0, tau=0.3)
    target = Entity(
        "target",
        tgt,
        tgt.initial_state([7000.0, 1200.0, 3000.0], [-700.0, 30.0, 0.0]),
        controller=None,
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=30.0,
    ).run()
    assert guidance.estimator_state.shape == (9,)  # 3-D [pos, vel, acc] belief
    assert res.miss_distance < 60.0  # noisy 3-D seeker still brings it home


def test_estimating_guidance_wraps_rl_controller():
    """ADR-0005 loop-closure: an RL-style controller fed a noisy-radar→EKF *estimate* still hits.

    Uses a PN-equivalent mock policy (no trained-model dependency) deployed via RLGuidance, wrapped
    in EstimatingGuidance — so the estimated target flows through the RL observation path."""
    from intercept.envs import POS_SCALE, VEL_SCALE
    from intercept.guidance.rl_policy import RLGuidance

    a_max = 300.0

    class _MockPNPolicy:
        # The observation is build_observation(...) = [r/POS, v_rel/VEL, v_own/VEL]; undo the
        # scaling and return the 1-D lateral PN command as a normalized action in [-1, 1].
        def predict(self, obs, deterministic=True):
            o = np.asarray(obs, dtype=float)
            r, v_rel = o[:2] * POS_SCALE, o[2:4] * VEL_SCALE
            rng = float(np.linalg.norm(r))
            if rng < 1e-6:
                return np.array([0.0], dtype=np.float32), None
            vc = -(r @ v_rel) / rng
            lam_dot = (r[0] * v_rel[1] - r[1] * v_rel[0]) / (rng * rng)
            return np.array([np.clip(4.0 * vc * lam_dot / a_max, -1.0, 1.0)], np.float32), None

    rng = np.random.default_rng(0)
    rl = RLGuidance("target", _MockPNPolicy(), a_max=a_max)
    guidance = EstimatingGuidance(
        "target",
        Radar(sigma_range=15.0, sigma_bearing=0.005),
        lambda x0, P0: EKF(lambda d: nca_model(d, 50.0), x0, P0),
        rl,
        rng,
    )
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, 700.0, 0.0]),
        controller=guidance,
        role="interceptor",
    )
    target = Entity("target", PointMass2D(), np.array([4000.0, 500.0, -250.0, 0.0]), role="target")
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=30.0,
        kill_radius=20.0,
    ).run()
    assert guidance.estimator is not None  # the estimator ran in the RL loop
    assert res.miss_distance < 40.0  # RL-via-estimate still brings it home


# --- INS platform error + 3-D IMM in the guidance loop -----------------------


def test_ins_error_bias_and_drift():
    from intercept.estimation import INSError

    ins = INSError(3, np.random.default_rng(0), bias_std=20.0, drift_rate=2.0)
    e0 = ins(0.0)
    assert e0.shape == (3,)
    assert np.allclose(e0, ins.bias)  # at t=0 the error is the bias
    assert np.allclose(ins(5.0), ins.bias + 5.0 * ins.drift)  # grows linearly with the drift


def test_3d_imm_in_loop_intercepts_and_ins_degrades():
    from intercept.core import Engagement, Entity, PointMass3D
    from intercept.core.aero import G0
    from intercept.estimation import INSError, make_cv_ca_imm
    from intercept.guidance import EstimatingGuidance, augmented_pn_3d
    from intercept.sensors import Radar3D

    i0 = np.array([0.0, 0.0, 0.0, 650.0, 0.0, 40.0])
    t0 = np.array([9000.0, 2200.0, 3200.0, -250.0, 30.0, 0.0])

    def run(platform_error):
        rng = np.random.default_rng(0)
        radar = Radar3D(sigma_range=20.0, sigma_az=0.004, sigma_el=0.004)
        guid = EstimatingGuidance(
            "target",
            radar,
            lambda x0, P0: make_cv_ca_imm(x0, P0, ndim=3),
            augmented_pn_3d("target", N=4.0),
            rng,
            platform_error=platform_error,
        )
        intc = Entity(
            "interceptor",
            PointMass3D(a_max=40 * G0),
            i0.copy(),
            controller=guid,
            role="interceptor",
        )
        tgt = Entity("target", PointMass3D(), t0.copy(), role="target")
        return Engagement(
            [intc, tgt],
            interceptor="interceptor",
            target="target",
            dt=0.01,
            t_max=30.0,
            kill_radius=25.0,
        ).run()

    clean = run(None)
    assert clean.intercepted  # 3-D IMM closes the loop and intercepts
    noisy = run(INSError(3, np.random.default_rng(7), bias_std=60.0, drift_rate=15.0))
    assert noisy.miss_distance > clean.miss_distance  # INS platform error degrades the result
