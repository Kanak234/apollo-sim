"""Phases 13 & 15 — orbital depots, ISRU, and Mars EDL.

depot:    cryogenic boil-off, refuelling ledger, direct-vs-depot
          architecture comparison (the honest version of the "petrol
          pump in space" idea — Lagrange/LEO depots, not fuel stations
          floating on the route).
isru:     Sabatier propellant production — CO2 + 4H2 -> CH4 + 2H2O,
          with electrolysis closing the loop. Power-limited production.
edl_mars: why Mars landing is the hardest EDL in the solar system —
          the atmosphere is thick enough to burn you, too thin to stop
          you. Parachute alone leaves ~60 m/s at the ground; a powered
          final phase is mandatory.
"""
from __future__ import annotations

import math

import engine
import numpy as np
from bodies import G0, MARS

# ---------------------------------------------------------------- depot
BOILOFF_PER_DAY = {          # fraction of remaining propellant per day
    "LH2_passive": 0.010,    # ~1 %/day, typical uninsulated estimate
    "LH2_cryocooled": 0.001, # active zero-boil-off-ish with power cost
    "CH4_passive": 0.0005,   # methane is far easier to keep
}


def propellant_after_storage(m0: float, days: float, kind: str) -> float:
    """Exponential boil-off: m(t) = m0 * (1 - rate)^days."""
    rate = BOILOFF_PER_DAY[kind]
    return m0 * (1.0 - rate) ** days


def stage_propellant_for_dv(payload: float, dv: float, isp: float,
                            dry_fraction: float = 0.10) -> float:
    """Propellant needed to push `payload` through `dv`, when the stage
    dry mass is dry_fraction * propellant. Solves the rocket equation
    with structure included."""
    ve = isp * G0
    mr = math.exp(dv / ve)
    denom = 1.0 - dry_fraction * (mr - 1.0)
    if denom <= 0.0:
        return math.inf
    return (mr - 1.0) * payload / denom


def architecture_compare(payload: float, dv_tmi: float, isp: float,
                         wait_days: float, kind: str):
    """Direct launch vs refuel-at-depot for the same TMI payload.

    direct:  everything (payload + full stage) on one launch.
    depot:   the stage launches dry-ish, propellant is delivered by
             tankers earlier and stored (paying boil-off), the crew
             launch is small.

    Returns both total IMLEO and the largest single launch each
    architecture demands — the second number is why depots matter.
    """
    prop = stage_propellant_for_dv(payload, dv_tmi, isp)
    dry = 0.10 * prop
    direct_total = payload + prop + dry
    # depot: tankers must deliver enough that after `wait_days` of
    # storage the required propellant remains
    prop_delivered = prop / ((1.0 - BOILOFF_PER_DAY[kind]) ** wait_days)
    depot_total = payload + prop_delivered + dry
    depot_biggest_launch = payload + dry          # crew + dry stage
    return {
        "prop_needed": prop,
        "direct_imleo": direct_total,
        "direct_biggest_launch": direct_total,
        "depot_imleo": depot_total,
        "depot_biggest_launch": depot_biggest_launch,
        "boiloff_penalty": prop_delivered - prop,
    }


# ----------------------------------------------------------------- isru
SABATIER_ENERGY_KWH_PER_KG = 15.0   # kWh per kg of CH4/O2 propellant,
                                    # electrolysis-dominated, MOXIE-class
                                    # efficiency assumptions


def isru_production_rate(power_kw: float) -> float:
    """kg of propellant per day at a given continuous power level."""
    return power_kw * 24.0 / SABATIER_ENERGY_KWH_PER_KG


def days_to_fill(tank_kg: float, power_kw: float) -> float:
    return tank_kg / isru_production_rate(power_kw)


# ------------------------------------------------------------- Mars EDL
LANDER_MASS = 600.0
LANDER_CDA = 1.7 * 5.5          # beta ~ 64 kg/m^2, Phoenix-class
CHUTE_CDA = 0.6 * 113.0         # 12 m disk-gap-band
ENTRY_ALT_MARS = 125_000.0


def fly_mars_edl(v_entry: float = 5_600.0, gamma_deg: float = -12.0,
                 powered_final: bool = True, dt: float = 0.05):
    """Mars entry -> chute -> (optional) powered final descent.

    Without the powered phase the lander hits at chute terminal
    velocity (~55-60 m/s): loss of mission. With it, a ~200 m/s
    delta-v landing burn brings touchdown under 3 m/s.
    """
    r0 = MARS.radius + ENTRY_ALT_MARS
    g = math.radians(gamma_deg)
    s = np.array([r0, 0.0, v_entry * math.sin(g), v_entry * math.cos(g),
                  LANDER_MASS])
    t = 0.0
    peak_g = 0.0
    chute_on = False
    burn_dv = 0.0
    ve = 300.0 * G0                    # storable-prop lander engine
    outcome = None

    while t < 2_000.0:
        r = math.hypot(s[0], s[1])
        alt = r - MARS.radius
        v = math.hypot(s[2], s[3])

        cda = LANDER_CDA
        if not chute_on and alt < 12_000.0 and v < 420.0:
            chute_on = True            # ~Mach 2 deploy
        if chute_on:
            cda = LANDER_CDA + CHUTE_CDA

        rho = MARS.density(alt)
        a_aero = 0.5 * rho * v * v * cda / s[4]
        peak_g = max(peak_g, a_aero / 9.80665)

        thrust = 0.0
        if powered_final and alt < 1_200.0:
            # constant-deceleration guidance to reach 1 m/s at the pad
            v_down = -(s[0] * s[2] + s[1] * s[3]) / r
            if v_down > 1.0:
                a_need = (v_down * v_down - 1.0) / (2.0 * max(alt, 1.0))
                g_loc = MARS.mu / (r * r)
                thrust = min(s[4] * (a_need + g_loc), 3.0 * s[4] * G0)
                burn_dv += thrust / s[4] * dt

        if alt <= 0.0:
            outcome = "landed" if v < 3.0 else "crashed"
            break

        def ctrl(tt, ss, thrust=thrust):
            rr = math.hypot(ss[0], ss[1])
            return (thrust, ss[0] / rr, ss[1] / rr)   # thrust straight up

        s = engine.rk4(t, s, dt, MARS, ctrl if thrust > 0 else None,
                       ve, cda, 0.0, 1.0, m_min=100.0)
        t += dt

    return {
        "outcome": outcome or "timeout", "t": t,
        "final_v": math.hypot(s[2], s[3]),
        "peak_g": peak_g, "burn_dv": burn_dv,
        "chute_deployed": chute_on,
    }
