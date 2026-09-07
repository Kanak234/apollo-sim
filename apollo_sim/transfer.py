"""Phases 9-10 — translunar injection, patched-conic arrival, LOI, TEI.

Method (classic patched conic, Curtis ch. 8 style, coplanar):
  1. TLI at LEO perigee sets a geocentric transfer ellipse.
  2. The ellipse is followed until it crosses the Moon's sphere of
     influence (SOI, 66,100 km), located by the arrival angle lambda
     measured at the Moon from the Earth-Moon line.
  3. Velocity is converted to the Moon frame (subtract the Moon's
     orbital velocity); the selenocentric conic gives the perilune.
  4. LOI is the impulsive burn at perilune from the arrival hyperbola
     onto the capture ellipse.

Design knobs: TLI speed v_p and arrival angle lambda. We solve for the
pair that hits a requested perilune altitude, preferring low TLI dv.
"""
from __future__ import annotations

import math

import kepler
from bodies import EARTH, EARTH_MOON_DIST, MOON

MOON_SOI = 6.61e7
V_MOON = math.sqrt(EARTH.mu / EARTH_MOON_DIST)   # 1,018 m/s circular


def tli_delta_v_hohmann(r_leo: float) -> float:
    """Minimum-energy TLI: transfer ellipse with apogee at lunar distance."""
    a = (r_leo + EARTH_MOON_DIST) / 2.0
    return kepler.vis_viva(EARTH.mu, r_leo, a) - kepler.circular_velocity(
        EARTH.mu, r_leo)


def _arrival_state(r_leo: float, v_p: float, lam: float):
    """Geocentric state at SOI crossing for TLI speed v_p, arrival angle lam.

    Returns dict or None if the transfer never reaches the SOI sphere.
    Geometry: r1 = distance Earth->SOI point, from the triangle
    Earth - Moon - SOI point with angle lam at the Moon.
    """
    r1 = math.sqrt(EARTH_MOON_DIST ** 2 + MOON_SOI ** 2
                   - 2.0 * EARTH_MOON_DIST * MOON_SOI * math.cos(lam))
    energy = 0.5 * v_p * v_p - EARTH.mu / r_leo
    a = -EARTH.mu / (2.0 * energy)
    e = 1.0 - r_leo / a                      # burn at perigee
    if a * (1.0 + e) < r1:                   # apogee below SOI point
        return None
    v1 = kepler.vis_viva(EARTH.mu, r1, a)
    # flight path angle from angular momentum conservation
    h = r_leo * v_p
    cos_g = h / (r1 * v1)
    cos_g = min(1.0, cos_g)
    gamma1 = math.acos(cos_g)                # >0, outbound
    # angle of the SOI point seen from Earth, from the same triangle
    sin_ang = MOON_SOI * math.sin(lam) / r1
    ang = math.asin(max(-1.0, min(1.0, sin_ang)))
    # time of flight perigee -> r1
    theta1 = kepler.theta_at_radius(a, e, r1)
    tof = kepler.time_from_periapsis(a, e, theta1, EARTH.mu)
    return {"r1": r1, "v1": v1, "gamma1": gamma1, "earth_angle": ang,
            "a": a, "e": e, "tof": tof}


def _selenocentric(arr: dict, lam: float):
    """Convert the SOI-crossing state to the Moon frame (Curtis 8.x).

    Standard planar geometry: v2 and its angle to the Moon-centred
    radius give the selenocentric conic and perilune.
    """
    v1, g1, ang = arr["v1"], arr["gamma1"], arr["earth_angle"]
    # velocity relative to the Moon
    v2 = math.sqrt(v1 * v1 + V_MOON * V_MOON
                   - 2.0 * v1 * V_MOON * math.cos(g1 - ang))
    # angle between selenocentric velocity and Moon-centred position
    # (epsilon2 in the standard derivation)
    sin_e2 = (V_MOON * math.cos(lam)
              - v1 * math.cos(lam + g1 - ang)) / v2
    sin_e2 = max(-1.0, min(1.0, sin_e2))
    eps2 = math.asin(sin_e2)
    # selenocentric orbit at r = SOI with speed v2, flight angle eps2
    energy = 0.5 * v2 * v2 - MOON.mu / MOON_SOI
    h = MOON_SOI * v2 * math.cos(eps2)
    a = math.inf if abs(energy) < 1e-9 else -MOON.mu / (2.0 * energy)
    e = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / MOON.mu ** 2))
    rp = h * h / MOON.mu / (1.0 + e)
    v_inf = math.sqrt(max(0.0, 2.0 * energy))
    return {"v2": v2, "rp": rp, "e": e, "a": a, "energy": energy, "v_inf": v_inf,
            "h": h}


def design_tli(r_leo: float, perilune_alt: float = 111_000.0):
    """Search (v_p, lambda) for a transfer hitting the target perilune.

    Coarse grid then local refine. Returns the full design dict.
    """
    target_rp = MOON.radius + perilune_alt
    v_hoh = kepler.circular_velocity(EARTH.mu, r_leo) + tli_delta_v_hohmann(
        r_leo)
    best = None
    for dv_extra in [x * 2.0 for x in range(60)]:          # 0..118 m/s
        v_p = v_hoh + dv_extra
        for lam_deg in range(5, 85, 2):
            lam = math.radians(lam_deg)
            arr = _arrival_state(r_leo, v_p, lam)
            if arr is None:
                continue
            sel = _selenocentric(arr, lam)
            err = sel["rp"] - target_rp
            score = abs(err)
            if best is None or score < best["score"]:
                best = {"score": score, "v_p": v_p, "lam": lam,
                        "arr": arr, "sel": sel}
        if best is not None and best["score"] < 3_000.0:
            break
    if best is None:
        raise RuntimeError("no lunar transfer found")
    # local refine on lambda
    v_p = best["v_p"]
    lo, hi = best["lam"] - math.radians(2), best["lam"] + math.radians(2)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        arr = _arrival_state(r_leo, v_p, mid)
        if arr is None:
            break
        sel = _selenocentric(arr, mid)
        if (sel["rp"] - target_rp) * (best["sel"]["rp"] - target_rp) > 0:
            lo = mid
        else:
            hi = mid
        if abs(sel["rp"] - target_rp) < abs(best["sel"]["rp"] - target_rp):
            best = {"score": abs(sel["rp"] - target_rp), "v_p": v_p,
                    "lam": mid, "arr": arr, "sel": sel}
        if best["score"] < 200.0:
            break
    v_circ = kepler.circular_velocity(EARTH.mu, r_leo)
    return {
        "tli_dv": best["v_p"] - v_circ,
        "lambda_deg": math.degrees(best["lam"]),
        "tof_days": best["arr"]["tof"] / 86_400.0,
        "perilune_alt": best["sel"]["rp"] - MOON.radius,
        "v_inf": best["sel"]["v_inf"],
        "arrival": best["sel"],
    }


def loi_delta_v(v_inf: float, perilune_alt: float = 111_000.0,
                apolune_alt: float = 314_000.0) -> float:
    """Capture burn at perilune: hyperbola -> 111 x 314 km ellipse."""
    rp = MOON.radius + perilune_alt
    ra = MOON.radius + apolune_alt
    v_hyp = math.sqrt(v_inf * v_inf + 2.0 * MOON.mu / rp)
    a_ell = (rp + ra) / 2.0
    v_ell = kepler.vis_viva(MOON.mu, rp, a_ell)
    return v_hyp - v_ell


def circ_dv(perilune_alt: float = 111_000.0,
            apolune_alt: float = 314_000.0) -> float:
    """LOI-2: circularise at perilune of the capture ellipse."""
    rp = MOON.radius + perilune_alt
    a_ell = (rp + MOON.radius + apolune_alt) / 2.0
    return kepler.vis_viva(MOON.mu, rp, a_ell) - kepler.circular_velocity(
        MOON.mu, rp)


def tei_delta_v(orbit_alt: float = 111_000.0,
                v_inf_return: float = 900.0) -> float:
    """Trans-Earth injection from circular lunar orbit.

    v_inf_return ~ 0.9 km/s puts the CSM on an Earth-return ellipse with
    perigee inside the entry corridor (the exact value depends on the
    return geometry; 0.8-1.0 km/s brackets the Apollo missions).
    """
    r = MOON.radius + orbit_alt
    v_esc_with_vinf = math.sqrt(v_inf_return ** 2 + 2.0 * MOON.mu / r)
    return v_esc_with_vinf - kepler.circular_velocity(MOON.mu, r)
