"""Phase 8 — Saturn V ascent: liftoff to 185 km parking orbit.

Three real stages, exponential atmosphere, gravity-turn pitch program,
apoapsis-targeted S-IVB insertion, and a delta-v loss ledger.

The headline lesson this module produces as numbers:
    orbital speed needed        ~ 7,800 m/s
    ideal delta-v spent         ~ 9,300-9,600 m/s
    the difference              = gravity + drag (+ steering) losses
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import engine
import kepler
import numpy as np
from bodies import EARTH, EARTH_ROT_SPEED_KSC, G0


@dataclass
class Stage:
    name: str
    thrust_sl: float      # N (sea level); 0 -> vacuum-only stage
    thrust_vac: float     # N
    isp_sl: float         # s
    isp_vac: float        # s
    prop: float           # kg
    dry: float            # kg

    def thrust_isp(self, altitude: float) -> tuple[float, float]:
        """Blend sea-level -> vacuum with ambient pressure ratio."""
        p = math.exp(-max(altitude, 0.0) / EARTH.atm_scale_h)
        if self.thrust_sl <= 0.0:
            return self.thrust_vac, self.isp_vac
        thrust = self.thrust_vac - (self.thrust_vac - self.thrust_sl) * p
        isp = self.isp_vac - (self.isp_vac - self.isp_sl) * p
        return thrust, isp


# Real Saturn V / Apollo 11 stage data (approximate flight values).
S_IC = Stage("S-IC", 3.34e7, 3.90e7, 263.0, 304.0, 2_077_000.0, 131_000.0)
S_II = Stage("S-II", 0.0, 5.00e6, 421.0, 421.0, 456_000.0, 36_000.0)
S_IVB = Stage("S-IVB", 0.0, 1.00e6, 421.0, 421.0, 108_000.0, 13_500.0)

PAYLOAD = 30_320.0 + 15_100.0 + 4_000.0   # CSM + LM + SLA/IU approx [kg]
CDA_STACK = 0.5 * math.pi * (10.06 / 2) ** 2   # Cd=0.5, D=10.06 m

TARGET_ALT = 185_000.0
PITCH_START_ALT = 100.0       # m — begin pitch-over
PITCH_S1_ALT = 35_500.0       # m — end of S-IC pitch ramp
PITCH_S1_MAX = math.radians(58.6)
PITCH_S2_ALT = 122_000.0      # m — end of S-II pitch ramp
PITCH_S2_MAX = math.radians(74.2)
PITCH_MAX = PITCH_S2_MAX


def _pitch_s1(alt: float) -> float:
    """Prescribed pitch program for S-IC boost phase."""
    if alt <= PITCH_START_ALT:
        return 0.0
    if alt <= PITCH_S1_ALT:
        return PITCH_S1_MAX * (alt - PITCH_START_ALT) / (PITCH_S1_ALT - PITCH_START_ALT)
    return PITCH_S1_MAX


def _pitch_s2(alt: float) -> float:
    """Prescribed pitch program for S-II boost phase."""
    if alt <= PITCH_S1_ALT:
        return PITCH_S1_MAX
    if alt <= PITCH_S2_ALT:
        return PITCH_S1_MAX + (PITCH_S2_MAX - PITCH_S1_MAX) * (alt - PITCH_S1_ALT) / (PITCH_S2_ALT - PITCH_S1_ALT)
    return PITCH_S2_MAX


def _pitch_from_vertical(alt: float) -> float:
    """Prescribed pitch program across boost phase."""
    if alt <= PITCH_S1_ALT:
        return _pitch_s1(alt)
    return _pitch_s2(alt)


def fly_to_orbit(dt: float = 0.2, record: bool = True):
    """Fly the full ascent. Returns a result dict with orbit + loss ledger."""
    r0 = EARTH.radius
    # Launch site at angle 0: radial = +x, east = +y. Earth rotation credit:
    s = np.array([r0, 0.0, 0.0, EARTH_ROT_SPEED_KSC,
                  S_IC.prop + S_IC.dry + S_II.prop + S_II.dry
                  + S_IVB.prop + S_IVB.dry + PAYLOAD])

    losses = {"gravity": 0.0, "drag": 0.0, "ideal_dv": 0.0}
    hist = []
    t = 0.0
    stage_log = []

    def step_burn(stage: Stage, m_after: float, pitch_fn):
        nonlocal s, t
        while s[4] > m_after + 1e-6:
            alt = math.hypot(s[0], s[1]) - EARTH.radius
            thrust, isp = stage.thrust_isp(alt)
            ve = isp * G0
            r = math.hypot(s[0], s[1])
            rx, ry = s[0] / r, s[1] / r
            ex, ey = -ry, rx            # local east (prograde launch)
            ph = pitch_fn(alt)
            ux = rx * math.cos(ph) + ex * math.sin(ph)
            uy = ry * math.cos(ph) + ey * math.sin(ph)

            def ctrl(tt, ss, t_val=thrust, ux_val=ux, uy_val=uy):
                return (t_val, ux_val, uy_val)

            g = EARTH.mu / (r * r)
            gamma = kepler.flight_path_angle(*s[:4])
            v = math.hypot(s[2], s[3])
            rho = EARTH.density(alt)
            a_drag = 0.5 * rho * v * v * CDA_STACK / s[4]
            losses["gravity"] += g * math.sin(max(gamma, 0.0)) * dt
            losses["drag"] += a_drag * dt
            losses["ideal_dv"] += thrust / s[4] * dt
            s = engine.rk4(t, s, dt, EARTH, ctrl, ve, CDA_STACK, m_min=m_after)
            t += dt
            if record and int(t / dt) % 25 == 0:
                hist.append((t, s.copy()))
        # jettison stage dry mass
        s[4] = max(s[4] - stage.dry, 0.0)
        stage_log.append((stage.name, t, math.hypot(s[0], s[1]) - EARTH.radius,
                          math.hypot(s[2], s[3])))

    # S-IC burn
    step_burn(S_IC, s[4] - S_IC.prop, _pitch_s1)

    # S-II burn
    step_burn(S_II, s[4] - S_II.prop, _pitch_s2)

    # S-IVB burn 1: closed-loop orbital insertion into parking orbit
    m_after_sivb = s[4] - S_IVB.prop
    target_rp = 170_050.0

    while s[4] > m_after_sivb + 1e-6:
        r = math.hypot(s[0], s[1])
        alt = r - EARTH.radius
        el = kepler.elements(*s[:4], EARTH.mu)
        if el["rp"] - EARTH.radius >= target_rp:
            break

        thrust, isp = S_IVB.thrust_isp(alt)
        ve = isp * G0
        rx, ry = s[0] / r, s[1] / r
        ex, ey = -ry, rx

        v_rad = (s[0] * s[2] + s[1] * s[3]) / r
        g_local = EARTH.mu / (r * r)
        v_horiz = (s[0] * s[3] - s[1] * s[2]) / r
        a_centrif = v_horiz * v_horiz / r

        a_rad_cmd = (g_local - a_centrif) + 0.015 * (TARGET_ALT - alt) - 0.25 * v_rad
        thrust_accel = thrust / s[4]
        sin_pitch = max(-0.95, min(0.95, a_rad_cmd / thrust_accel))
        cos_pitch = math.sqrt(1.0 - sin_pitch * sin_pitch)
        ux = rx * sin_pitch + ex * cos_pitch
        uy = ry * sin_pitch + ey * cos_pitch

        gamma = kepler.flight_path_angle(*s[:4])
        losses["gravity"] += g_local * math.sin(max(gamma, 0.0)) * dt
        losses["ideal_dv"] += thrust / s[4] * dt

        def ctrl(tt, ss, t_val=thrust, ux_val=ux, uy_val=uy):
            return (t_val, ux_val, uy_val)

        s = engine.rk4(t, s, dt, EARTH, ctrl, ve, CDA_STACK, m_min=m_after_sivb)
        t += dt
        if record and int(t / dt) % 25 == 0:
            hist.append((t, s.copy()))

    stage_log.append(("S-IVB MECO-1", t,
                      math.hypot(s[0], s[1]) - EARTH.radius,
                      math.hypot(s[2], s[3])))

    el = kepler.elements(*s[:4], EARTH.mu)
    return {
        "t": t, "state": s, "elements": el, "losses": losses,
        "prop_left_sivb": s[4] - m_after_sivb,
        "stage_log": stage_log, "history": hist,
        "perigee_alt": el["rp"] - EARTH.radius,
        "apogee_alt": el["ra"] - EARTH.radius,
        "v": math.hypot(s[2], s[3]),
    }
