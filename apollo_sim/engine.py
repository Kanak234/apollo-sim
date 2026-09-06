"""Body-centred 2D propagation engine with RK4.

State vector: s = [x, y, vx, vy, m]  (body-centred inertial, metres, kg)

Forces:
    gravity  a = -mu * r_vec / |r|^3          (always)
    thrust   a = T/m along a commanded unit vector; dm/dt = -T/ve
    drag     a = -0.5 rho v^2 Cd A / m  (v-hat) when the body has an
             atmosphere and a CdA is supplied
    lift     a = (L/D)*|a_drag| perpendicular to velocity (bank 0 =
             in-plane lift-up), used for entry capsules

The control callback decides thrust each step:
    control(t, s) -> (thrust_N, ux, uy)
ux, uy need not be normalised; (0,0) means thrust along velocity.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from bodies import Body

Control = Callable[[float, np.ndarray], tuple]


def _accel(t: float, s: np.ndarray, body: Body, control: Optional[Control],
           ve: float, cda: float, lod: float, lift_sign: float) -> np.ndarray:
    x, y, vx, vy, m = s
    r = math.hypot(x, y)
    inv_r3 = 1.0 / (r * r * r)
    ax = -body.mu * x * inv_r3
    ay = -body.mu * y * inv_r3
    dm = 0.0

    if control is not None and m > 0.0:
        thrust, ux, uy = control(t, s)
        if thrust > 0.0:
            n = math.hypot(ux, uy)
            if n < 1e-12:                       # default: along velocity
                n = math.hypot(vx, vy)
                ux, uy = (vx / n, vy / n) if n > 1e-9 else (x / r, y / r)
            else:
                ux, uy = ux / n, uy / n
            ax += thrust / m * ux
            ay += thrust / m * uy
            dm = -thrust / ve

    if cda > 0.0 and body.atm_rho0 > 0.0:
        alt = r - body.radius
        rho = body.density(alt)
        if rho > 0.0:
            v = math.hypot(vx, vy)
            if v > 1e-6:
                a_d = 0.5 * rho * v * v * cda / m
                ax += -a_d * vx / v
                ay += -a_d * vy / v
                if lod > 0.0:
                    # in-plane lift, perpendicular to velocity
                    a_l = lod * a_d * lift_sign
                    ax += -a_l * vy / v * -1.0
                    ay += -a_l * vx / v
                    # (-vy, vx)/v is the +90 deg rotation of v-hat;
                    # lift_sign=+1 gives "lift up" for prograde entry
    return np.array([vx, vy, ax, ay, dm])


def rk4(t: float, s: np.ndarray, dt: float, body: Body,
        control: Optional[Control] = None, ve: float = 1.0,
        cda: float = 0.0, lod: float = 0.0, lift_sign: float = 1.0,
        m_min: float = 0.0) -> np.ndarray:
    """One RK4 step. Mass is clamped at m_min (dry mass)."""
    def f(tt, ss):
        return _accel(tt, ss, body, control, ve, cda, lod, lift_sign)

    k1 = f(t, s)
    k2 = f(t + dt / 2, s + dt / 2 * k1)
    k3 = f(t + dt / 2, s + dt / 2 * k2)
    k4 = f(t + dt, s + dt * k3)
    out = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    out[4] = max(out[4], m_min)
    return out


def propagate(s: np.ndarray, t0: float, t_max: float, dt: float, body: Body,
              control: Optional[Control] = None, ve: float = 1.0,
              cda: float = 0.0, lod: float = 0.0, lift_sign: float = 1.0,
              m_min: float = 0.0,
              stop: Optional[Callable[[float, np.ndarray], bool]] = None,
              record_every: int = 0):
    """Propagate until t_max or stop(t, s) is True.

    Returns (t, s, history) where history is a list of (t, s.copy())
    sampled every `record_every` steps (empty if 0).
    """
    t = t0
    hist = []
    steps = int(round((t_max - t0) / dt))
    for i in range(steps):
        if stop is not None and stop(t, s):
            break
        s = rk4(t, s, dt, body, control, ve, cda, lod, lift_sign, m_min)
        t += dt
        if record_every and i % record_every == 0:
            hist.append((t, s.copy()))
    return t, s, hist
