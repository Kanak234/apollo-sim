"""Phase 14 — Lambert's problem, launch windows, porkchop data.

Lambert: universal-variables formulation (Curtis, Algorithm 5.2).
Ephemerides: circular, coplanar planets — Earth at 1 AU, Mars at 1.524 AU.
That is deliberately simplified and clearly labelled: real mission design
adds eccentricity and 1.85 deg of inclination, which shifts numbers by a
few percent but changes nothing structurally.
"""
from __future__ import annotations

import math

import kepler
import numpy as np
from bodies import EARTH, MARS, SUN, SUN_EARTH_DIST, SUN_MARS_DIST

N_E = math.sqrt(SUN.mu / SUN_EARTH_DIST ** 3)     # rad/s
N_M = math.sqrt(SUN.mu / SUN_MARS_DIST ** 3)
V_E = N_E * SUN_EARTH_DIST
V_M = N_M * SUN_MARS_DIST


def synodic_period_days() -> float:
    return 2.0 * math.pi / (N_E - N_M) / 86_400.0


def earth_state(t: float):
    th = N_E * t
    return (np.array([math.cos(th), math.sin(th)]) * SUN_EARTH_DIST,
            np.array([-math.sin(th), math.cos(th)]) * V_E)


def mars_state(t: float, phase0: float):
    th = N_M * t + phase0
    return (np.array([math.cos(th), math.sin(th)]) * SUN_MARS_DIST,
            np.array([-math.sin(th), math.cos(th)]) * V_M)


# ------------------------------------------------------------- Lambert
def _stumpff_c(z: float) -> float:
    if z > 1e-8:
        return (1.0 - math.cos(math.sqrt(z))) / z
    if z < -1e-8:
        return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)
    return 0.5


def _stumpff_s(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz ** 3
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz ** 3
    return 1.0 / 6.0


def lambert(r1v: np.ndarray, r2v: np.ndarray, tof: float, mu: float,
            prograde: bool = True):
    """Solve Lambert's problem. Returns (v1, v2) or None if no solution.

    Universal-variable iteration on z (Curtis Alg. 5.2), planar.
    """
    r1, r2 = np.linalg.norm(r1v), np.linalg.norm(r2v)
    cross = r1v[0] * r2v[1] - r1v[1] * r2v[0]
    cos_dth = float(np.dot(r1v, r2v) / (r1 * r2))
    cos_dth = max(-1.0, min(1.0, cos_dth))
    dth = math.acos(cos_dth)
    if prograde and cross < 0.0:
        dth = 2.0 * math.pi - dth
    if not prograde and cross >= 0.0:
        dth = 2.0 * math.pi - dth

    A = math.sin(dth) * math.sqrt(r1 * r2 / (1.0 - math.cos(dth)))
    if abs(A) < 1e-9:
        return None

    def y(z):
        cz = _stumpff_c(z)
        if cz <= 1e-12:
            return -1.0
        return r1 + r2 + A * (z * _stumpff_s(z) - 1.0) / math.sqrt(cz)

    def F(z):
        cz = _stumpff_c(z)
        if cz <= 1e-12:
            return None
        yy = y(z)
        if yy < 0.0:
            return None
        return ((yy / cz) ** 1.5 * _stumpff_s(z)
                + A * math.sqrt(yy) - math.sqrt(mu) * tof)

    # bracket the root in z
    z_lo, z_hi = -4.0 * math.pi, 4.0 * math.pi ** 2 - 0.1
    f_lo = F(z_lo)
    while f_lo is None and z_lo < z_hi:
        z_lo += 1.0
        f_lo = F(z_lo)
    f_hi = F(z_hi)
    while f_hi is None and z_hi > z_lo:
        z_hi -= 0.5
        f_hi = F(z_hi)
    if f_lo is None or f_hi is None or f_lo * f_hi > 0.0:
        return None
    for _ in range(120):
        z = 0.5 * (z_lo + z_hi)
        f = F(z)
        if f is None:
            z_hi = z
            continue
        if abs(f) < 1e-6:
            break
        if f_lo * f < 0.0:
            z_hi = z
        else:
            z_lo, f_lo = z, f

    yy = y(z)
    f_l = 1.0 - yy / r1
    g_l = A * math.sqrt(yy / mu)
    gdot = 1.0 - yy / r2
    v1 = (r2v - f_l * r1v) / g_l
    v2 = (gdot * r2v - r1v) / g_l
    return v1, v2


# ------------------------------------------------------------- windows
def hohmann_numbers():
    """Reference Hohmann Earth->Mars values (heliocentric, circular)."""
    a = (SUN_EARTH_DIST + SUN_MARS_DIST) / 2.0
    v_dep = kepler.vis_viva(SUN.mu, SUN_EARTH_DIST, a)
    v_arr = kepler.vis_viva(SUN.mu, SUN_MARS_DIST, a)
    tof = math.pi * math.sqrt(a ** 3 / SUN.mu)
    return {
        "v_inf_dep": v_dep - V_E,
        "v_inf_arr": V_M - v_arr,
        "tof_days": tof / 86_400.0,
        "phase_deg": math.degrees(math.pi - N_M * tof),
    }


def tmi_dv_from_leo(v_inf: float, r_leo: float) -> float:
    v_p = math.sqrt(v_inf * v_inf + 2.0 * EARTH.mu / r_leo)
    return v_p - kepler.circular_velocity(EARTH.mu, r_leo)


def capture_dv_at_mars(v_inf: float, r_orbit: float) -> float:
    v_p = math.sqrt(v_inf * v_inf + 2.0 * MARS.mu / r_orbit)
    return v_p - kepler.circular_velocity(MARS.mu, r_orbit)


def porkchop(phase0: float, dep_days, tof_days, r_leo: float,
             r_mars_orbit: float):
    """Total mission dv over a (departure x time-of-flight) grid.

    Returns a 2D numpy array of dv [m/s]; np.nan where Lambert fails.
    """
    out = np.full((len(dep_days), len(tof_days)), np.nan)
    for i, dd in enumerate(dep_days):
        t_dep = dd * 86_400.0
        rE, vE = earth_state(t_dep)
        for j, td in enumerate(tof_days):
            tof = td * 86_400.0
            rM, vM = mars_state(t_dep + tof, phase0)
            sol = lambert(rE, rM, tof, SUN.mu)
            if sol is None:
                continue
            v1, v2 = sol
            vinf_d = float(np.linalg.norm(v1 - vE))
            vinf_a = float(np.linalg.norm(v2 - vM))
            out[i, j] = (tmi_dv_from_leo(vinf_d, r_leo)
                         + capture_dv_at_mars(vinf_a, r_mars_orbit))
    return out
