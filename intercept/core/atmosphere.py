"""Standard atmosphere (ISA) and a transonic drag model — the physical basis for L3 realism.

Real missile performance is set by the air it flies through: drag and the maximum aerodynamic turn
(lift) both scale with **dynamic pressure** ``q = ½ρV²``, which falls with altitude and speed, and
drag rises sharply through the transonic region. Modeling these from physics (rather than fixed
constants) is what makes the L3 dynamics physically grounded — turn capability and energy loss
*emerge* from the flight condition instead of being prescribed.

* :func:`isa` — International Standard Atmosphere (troposphere + lower stratosphere, to ~20 km):
  density, speed of sound, pressure, temperature vs. geometric altitude.
* :func:`drag_coefficient` — a representative zero-lift drag coefficient ``Cd0(Mach)`` with the
  characteristic transonic drag rise and supersonic relaxation.
"""

from __future__ import annotations

import numpy as np

# ISA constants
_G0 = 9.80665  # m/s²
_R = 287.05  # J/(kg·K) specific gas constant for air
_GAMMA = 1.4  # ratio of specific heats
_T0 = 288.15  # K  sea-level temperature
_P0 = 101325.0  # Pa sea-level pressure
_L = 0.0065  # K/m tropospheric lapse rate
_H_TROP = 11000.0  # m  tropopause
_T_TROP = _T0 - _L * _H_TROP  # 216.65 K
_P_TROP = _P0 * (_T_TROP / _T0) ** (_G0 / (_R * _L))


def isa(altitude: float) -> tuple[float, float, float, float]:
    """ISA properties at geometric ``altitude`` (m): ``(density, sound_speed, pressure, temp)``.

    Valid to ~20 km (troposphere + lower stratosphere); altitude is clamped to ``≥ 0``.
    """
    h = max(float(altitude), 0.0)
    if h < _H_TROP:
        temp = _T0 - _L * h
        pressure = _P0 * (temp / _T0) ** (_G0 / (_R * _L))
    else:
        temp = _T_TROP
        pressure = _P_TROP * np.exp(-_G0 * (h - _H_TROP) / (_R * _T_TROP))
    density = pressure / (_R * temp)
    sound_speed = float(np.sqrt(_GAMMA * _R * temp))
    return density, sound_speed, pressure, temp


def density(altitude: float) -> float:
    """Air density (kg/m³) at altitude."""
    return isa(altitude)[0]


def mach(speed: float, altitude: float) -> float:
    """Mach number for ``speed`` (m/s) at ``altitude`` (m)."""
    return speed / isa(altitude)[1]


def drag_coefficient(mach_number: float) -> float:
    """Representative zero-lift drag coefficient ``Cd0`` vs Mach (transonic rise, supersonic decay).

    Base subsonic ~0.20 with a transonic bump peaking ~0.55 near M≈1.1 — qualitatively the standard
    drag-divergence behavior. Representative (not vehicle-specific); see ADR-0008.
    """
    m = abs(float(mach_number))
    return 0.20 + 0.35 * float(np.exp(-(((m - 1.1) / 0.30) ** 2)))
