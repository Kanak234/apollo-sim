"""Phase 11 — lunar ascent and Clohessy-Wiltshire rendezvous.

CW frame convention (target-centred LVLH):
    x = radial (up, away from the Moon)
    y = along-track (direction of target motion)
    n = target mean motion = sqrt(mu / a^3)

CW equations (Hill's equations) — the linearised relative dynamics:
    x'' - 3 n^2 x - 2 n y' = 0
    y'' + 2 n x'           = 0

The famous counter-intuitive result lives in these equations: a burn
that LOWERS your orbit (x < 0) makes you drift FORWARD (y grows),
because lower orbits are faster. To catch a target ahead of you,
you slow down.
"""
from __future__ import annotations

import math

import engine
import kepler
import numpy as np
from bodies import G0, MOON


# ---------------------------------------------------------------- CW core
def cw_matrices(n: float, t: float):
    """State transition blocks for the CW equations at time t.

    [r(t)]   [Mrr Mrv] [r0]
    [v(t)] = [Mvr Mvv] [v0]      with r=(x,y), v=(x',y')
    """
    s, c = math.sin(n * t), math.cos(n * t)
    Mrr = np.array([[4.0 - 3.0 * c, 0.0],
                    [6.0 * (s - n * t), 1.0]])
    Mrv = np.array([[s / n, 2.0 * (1.0 - c) / n],
                    [2.0 * (c - 1.0) / n, (4.0 * s - 3.0 * n * t) / n]])
    Mvr = np.array([[3.0 * n * s, 0.0],
                    [6.0 * n * (c - 1.0), 0.0]])
    Mvv = np.array([[c, 2.0 * s],
                    [-2.0 * s, 4.0 * c - 3.0]])
    return Mrr, Mrv, Mvr, Mvv


def cw_propagate(r0, v0, n: float, t: float):
    Mrr, Mrv, Mvr, Mvv = cw_matrices(n, t)
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    return Mrr @ r0 + Mrv @ v0, Mvr @ r0 + Mvv @ v0


def two_impulse_intercept(r0, v0, n: float, t_f: float):
    """Solve the two-impulse rendezvous: arrive at the target (r=0) at t_f.

    dv1 puts the chaser on the intercept path; dv2 nulls the arrival
    velocity. Returns (dv1_vec, dv2_vec, total_dv).
    """
    Mrr, Mrv, Mvr, Mvv = cw_matrices(n, t_f)
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    v0_req = np.linalg.solve(Mrv, -Mrr @ r0)
    dv1 = v0_req - v0
    v_arr = Mvr @ r0 + Mvv @ v0_req
    dv2 = -v_arr
    return dv1, dv2, float(np.linalg.norm(dv1) + np.linalg.norm(dv2))


# ---------------------------------------------------------------- ascent
APS_THRUST = 15_600.0     # N
APS_ISP = 311.0           # s
ASC_MASS_WET = 4_700.0    # kg at lunar liftoff
ASC_PROP = 2_353.0        # kg usable APS propellant

TARGET_PERILUNE = 18_000.0    # m — the real insertion orbit was ~17 x 84 km
TARGET_APOLUNE = 84_000.0


def fly_ascent(dt: float = 0.1):
    """Fly the LM ascent stage from the surface to lunar orbit.

    Profile: vertical rise 10 s, pitch to 52 deg off-vertical until
    the vertical rate is captured, then thrust along local horizontal
    until the orbit's perilune and apolune both exceed target.
    """
    r0 = MOON.radius
    s = np.array([r0, 0.0, 0.0, 0.0, ASC_MASS_WET])
    m_dry = ASC_MASS_WET - ASC_PROP
    ve = APS_ISP * G0
    t = 0.0
    dv_ideal = 0.0

    def orbit_ok(ss):
        el = kepler.elements(*ss[:4], MOON.mu)
        return (el["rp"] - MOON.radius >= TARGET_PERILUNE
                and el["ra"] != math.inf
                and el["ra"] - MOON.radius >= TARGET_APOLUNE)

    def ctrl(tt, ss):
        if ss[4] <= m_dry:
            return (0.0, 0.0, 0.0)
        r = math.hypot(ss[0], ss[1])
        rx, ry = ss[0] / r, ss[1] / r
        ex, ey = -ry, rx
        if tt < 10.0:                       # vertical rise
            return (APS_THRUST, rx, ry)
        alt = r - MOON.radius
        if alt < 15_000.0:                  # pitched ascent
            ph = math.radians(52.0)
            return (APS_THRUST,
                    rx * math.cos(ph) + ex * math.sin(ph),
                    ry * math.cos(ph) + ey * math.sin(ph))
        return (APS_THRUST, ex, ey)         # horizontal push to orbit

    while t < 900.0 and s[4] > m_dry + 0.5 and not orbit_ok(s):
        dv_ideal += APS_THRUST / s[4] * dt
        s = engine.rk4(t, s, dt, MOON, ctrl, ve, m_min=m_dry)
        t += dt

    el = kepler.elements(*s[:4], MOON.mu)
    return {
        "t": t, "state": s, "dv_ideal": dv_ideal,
        "perilune_alt": el["rp"] - MOON.radius,
        "apolune_alt": (el["ra"] - MOON.radius) if el["ra"] != math.inf
        else math.inf,
        "prop_left": s[4] - m_dry,
        "insertion_ok": orbit_ok(s),
    }
