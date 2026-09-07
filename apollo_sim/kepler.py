"""Two-body orbit utilities: elements, vis-viva, anomalies, time of flight.

Everything is planar (2D). State is body-centred inertial Cartesian.
"""
from __future__ import annotations

import math


def vis_viva(mu: float, r: float, a: float) -> float:
    """Orbital speed at radius r on an orbit of semi-major axis a."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)


def period(mu: float, a: float) -> float:
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def elements(x: float, y: float, vx: float, vy: float, mu: float) -> dict:
    """Classical elements from a planar Cartesian state.

    Returns a, e, rp (periapsis radius), ra (apoapsis radius),
    specific energy, specific angular momentum h (signed, z-component),
    and true anomaly theta.
    """
    r = math.hypot(x, y)
    v2 = vx * vx + vy * vy
    energy = 0.5 * v2 - mu / r
    h = x * vy - y * vx                      # z-component of r x v
    if abs(energy) < 1e-12:
        a = math.inf
    else:
        a = -mu / (2.0 * energy)
    # eccentricity vector (planar)
    ex = (v2 / mu - 1.0 / r) * x - (x * vx + y * vy) / mu * vx
    ey = (v2 / mu - 1.0 / r) * y - (x * vx + y * vy) / mu * vy
    e = math.hypot(ex, ey)
    if e > 1e-12:
        cos_th = (ex * x + ey * y) / (e * r)
        cos_th = max(-1.0, min(1.0, cos_th))
        theta = math.acos(cos_th)
        if (x * vx + y * vy) < 0.0:
            theta = 2.0 * math.pi - theta
    else:
        theta = math.atan2(y, x)
    rp = a * (1.0 - e) if a is not math.inf else r
    ra = a * (1.0 + e) if (a is not math.inf and e < 1.0) else math.inf
    return {"a": a, "e": e, "rp": rp, "ra": ra, "energy": energy,
            "h": h, "theta": theta}


def radius_at_theta(a: float, e: float, theta: float) -> float:
    p = a * (1.0 - e * e)
    return p / (1.0 + e * math.cos(theta))


def theta_at_radius(a: float, e: float, r: float) -> float:
    """True anomaly (0..pi, outbound) at radius r on an ellipse."""
    p = a * (1.0 - e * e)
    c = (p / r - 1.0) / e
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def time_from_periapsis(a: float, e: float, theta: float, mu: float) -> float:
    """Kepler time from periapsis passage to true anomaly theta (ellipse)."""
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(theta / 2.0),
                         math.sqrt(1.0 + e) * math.cos(theta / 2.0))
    if E < 0.0:
        E += 2.0 * math.pi
    M = E - e * math.sin(E)
    return M * math.sqrt(a ** 3 / mu)


def flight_path_angle(x: float, y: float, vx: float, vy: float) -> float:
    """Angle of velocity above local horizontal, radians. >0 climbing."""
    r = math.hypot(x, y)
    v = math.hypot(vx, vy)
    if r == 0.0 or v == 0.0:
        return 0.0
    radial = (x * vx + y * vy) / r
    s = max(-1.0, min(1.0, radial / v))
    return math.asin(s)
