"""Phase 12 — Earth entry at lunar-return speed.

Command Module: m = 5,560 kg, Cd = 1.3, A = 11.95 m^2 (beta ~ 358 kg/m^2),
L/D = 0.29 flown lift-up (bank 0) in-plane.

The corridor: entry flight-path angle -6.5 deg +/- ~1 deg.
  Too shallow -> the capsule skips back out of the atmosphere.
  Too steep   -> peak deceleration exceeds crew limits.

Heating indicator: Sutton-Graves stagnation heat rate
  qdot = k * sqrt(rho / r_nose) * v^3,   k_Earth = 1.7415e-4 (SI)
"""
from __future__ import annotations

import math

import engine
import numpy as np
from bodies import EARTH

CM_MASS = 5_560.0
CM_CDA = 1.3 * 11.95
CM_LOD = 0.29
NOSE_R = 4.69                      # m, CM heat-shield effective radius
SG_K = 1.7415e-4

ENTRY_ALT = 121_900.0              # entry interface
DROGUE_ALT = 7_300.0
MAIN_ALT = 3_000.0
CHUTE_CDA_DROGUE = 0.8 * 60.0
CHUTE_CDA_MAIN = 0.8 * 1_540.0     # three mains, ~8.5 m/s splashdown


def fly_entry(v_entry: float = 11_032.0, gamma_deg: float = -6.5,
              lift_up: bool = True, dt: float = 0.05):
    """Integrate from entry interface to splashdown / skip-out / overload.

    Returns dict with outcome in {"splashdown", "skip", "overload"} plus
    peak g, peak heating, and the trajectory summary.
    """
    r0 = EARTH.radius + ENTRY_ALT
    gamma = math.radians(gamma_deg)
    # start at angle 0, moving prograde (+y) with the given descent angle
    s = np.array([r0, 0.0,
                  v_entry * math.sin(gamma),     # radial (negative = down)
                  v_entry * math.cos(gamma),
                  CM_MASS])
    t = 0.0
    peak_g = 0.0
    peak_q = 0.0
    outcome = None
    lift_sign = 1.0 if lift_up else 0.0
    downrange0 = math.atan2(s[1], s[0])

    while t < 3_000.0:
        r = math.hypot(s[0], s[1])
        alt = r - EARTH.radius
        v = math.hypot(s[2], s[3])

        cda = CM_CDA
        lod = CM_LOD * lift_sign
        chutes_active = False
        if alt < DROGUE_ALT and v < 220.0:
            cda = CM_CDA + CHUTE_CDA_DROGUE
            lod = 0.0
            chutes_active = True
        if alt < MAIN_ALT and v < 120.0:
            cda = CM_CDA + CHUTE_CDA_MAIN
            lod = 0.0
            chutes_active = True

        rho = EARTH.density(alt)
        a_aero = 0.5 * rho * v * v * cda / s[4] * math.sqrt(1.0 + lod * lod)
        g_sensed = a_aero / 9.80665
        if not chutes_active:
            peak_g = max(peak_g, g_sensed)
        qdot = SG_K * math.sqrt(max(rho, 0.0) / NOSE_R) * v ** 3
        peak_q = max(peak_q, qdot)

        if alt <= 0.0:
            outcome = "splashdown"
            break
        if alt > ENTRY_ALT + 5_000.0 and s[0] * s[2] + s[1] * s[3] > 0.0 and v > 7_900.0:
            outcome = "skip"
            break
        if g_sensed > 12.0 and not chutes_active:
            outcome = "overload"
            break

        s = engine.rk4(t, s, dt, EARTH, None, 1.0, cda, lod, 1.0)
        t += dt

    if outcome is None:
        outcome = "timeout"
    downrange = (math.atan2(s[1], s[0]) - downrange0) * EARTH.radius
    return {
        "outcome": outcome, "t": t, "peak_g": peak_g,
        "peak_heat_wm2": peak_q,
        "final_v": math.hypot(s[2], s[3]),
        "final_alt": math.hypot(s[0], s[1]) - EARTH.radius,
        "downrange_km": downrange / 1000.0,
    }
