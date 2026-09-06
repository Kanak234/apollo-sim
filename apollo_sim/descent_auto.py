"""Automated lunar powered descent — P63/P65-style, central gravity.

This closes the loop the interactive game cannot: it proves the full
descent from the 15.24 km descent-orbit perilune to a soft touchdown
works under CORRECT orbital gravity, with the real LM-5 numbers.

Guidance (simplified but honest):
  Braking:  full thrust retrograde until horizontal velocity is nearly
            killed. This is what P63 mostly is.
  Approach: pitch toward vertical, thrust modulated to fly the altitude-
            vs-sink-rate profile.
  Landing:  P65-style vertical phase — target sink rate ~1 m/s at
            contact, throttle from a proportional law.

Success = touchdown with |v_vertical| <= 3 m/s, |v_horiz| <= 1.2 m/s
(the game's own landing limits), with propellant to spare.
"""
from __future__ import annotations

import math

import numpy as np

from bodies import MOON, G0
import engine
import kepler

LM_WET = 15_100.0
LM_PROP = 8_200.0
DPS_MAX = 45_040.0
DPS_ISP = 311.0
PDI_ALT = 15_240.0


def _local_frame(s):
    r = math.hypot(s[0], s[1])
    rx, ry = s[0] / r, s[1] / r          # up
    ex, ey = -ry, rx                     # prograde tangent at start side
    return r, rx, ry, ex, ey


def fly_descent(dt: float = 0.05):
    r0 = MOON.radius + PDI_ALT
    v0 = kepler.vis_viva(MOON.mu, r0,
                         (r0 + MOON.radius + 110_000.0) / 2.0)  # 15x110 orbit
    s = np.array([r0, 0.0, 0.0, v0, LM_WET])
    m_dry = LM_WET - LM_PROP
    ve = DPS_ISP * G0
    t = 0.0
    dv_used = 0.0
    phase = "BRAKING"

    def ctrl_factory():
        def ctrl(tt, ss):
            nonlocal phase
            r, rx, ry, _, _ = _local_frame(ss)
            alt = r - MOON.radius
            # velocity split into vertical (radial) and horizontal
            v_up = (ss[0] * ss[2] + ss[1] * ss[3]) / r
            hx, hy = ss[2] - v_up * rx, ss[3] - v_up * ry
            v_h = math.hypot(hx, hy)

            if ss[4] <= m_dry:
                return (0.0, 0.0, 0.0)

            if phase == "BRAKING":
                if v_h < 15.0 or alt < 2_500.0:
                    phase = "APPROACH"
                else:
                    return (DPS_MAX, -hx, -hy)     # full retrograde

            if phase == "APPROACH":
                if alt < 150.0 and v_h < 3.0:
                    phase = "LANDING"
                else:
                    # kill remaining horizontal velocity + control sink
                    sink_target = -max(3.0, alt / 40.0)    # m/s downward
                    g_loc = MOON.mu / (r * r)
                    a_up = g_loc + 0.6 * (sink_target - v_up)
                    a_h = 0.8 * v_h
                    ax = a_up * rx + (-hx / max(v_h, 0.1)) * a_h
                    ay = a_up * ry + (-hy / max(v_h, 0.1)) * a_h
                    a_mag = math.hypot(ax, ay)
                    thrust = min(DPS_MAX, ss[4] * a_mag)
                    return (thrust, ax, ay)

            # LANDING: vertical, proportional sink control to 1 m/s
            g_loc = MOON.mu / (r * r)
            sink_target = -1.0 if alt < 30.0 else -min(3.0, alt / 15.0)
            a_up = g_loc + 1.0 * (sink_target - v_up)
            thrust = max(0.0, min(DPS_MAX, ss[4] * a_up))
            return (thrust, rx, ry)
        return ctrl

    ctrl = ctrl_factory()
    hist = []
    while t < 1_200.0:
        r = math.hypot(s[0], s[1])
        if r <= MOON.radius:
            break
        thrust, ux, uy = ctrl(t, s)
        if thrust > 0.0:
            dv_used += thrust / s[4] * dt
        s = engine.rk4(t, s, dt, MOON, ctrl, ve, m_min=m_dry)
        t += dt
        if int(t / dt) % 200 == 0:
            hist.append((t, s.copy(), phase))

    r, rx, ry, _, _ = _local_frame(s)
    v_up = (s[0] * s[2] + s[1] * s[3]) / r
    hx, hy = s[2] - v_up * rx, s[3] - v_up * ry
    v_h = math.hypot(hx, hy)
    ok = abs(v_up) <= 3.0 and v_h <= 1.2 and s[4] > m_dry
    return {
        "t": t, "v_vertical": v_up, "v_horizontal": v_h,
        "prop_left": s[4] - m_dry, "dv_used": dv_used,
        "landed_ok": ok, "phase": phase, "history": hist,
    }
