"""Core checks: the engine must reproduce analytical two-body results."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import kepler
import numpy as np
import pytest
from bodies import EARTH, MOON


class TestOrbits:
    def test_earth_circular_orbit_holds(self):
        r = EARTH.radius + 185_000.0
        v = kepler.circular_velocity(EARTH.mu, r)
        s = np.array([r, 0.0, 0.0, v, 1000.0])
        T = kepler.period(EARTH.mu, r)
        _, s, _ = engine.propagate(s, 0.0, T, 1.0, EARTH)
        assert math.hypot(s[0], s[1]) == pytest.approx(r, abs=2.0)

    def test_energy_conservation_one_orbit(self):
        r = MOON.radius + 15_240.0
        v = kepler.circular_velocity(MOON.mu, r)
        s = np.array([r, 0.0, 0.0, v, 1000.0])
        e0 = 0.5 * v * v - MOON.mu / r
        T = kepler.period(MOON.mu, r)
        _, s, _ = engine.propagate(s, 0.0, T, 0.5, MOON)
        rr = math.hypot(s[0], s[1])
        e1 = 0.5 * (s[2] ** 2 + s[3] ** 2) - MOON.mu / rr
        assert e1 == pytest.approx(e0, rel=1e-10)

    def test_elements_roundtrip(self):
        r = EARTH.radius + 185_000.0
        v = 1.05 * kepler.circular_velocity(EARTH.mu, r)
        el = kepler.elements(r, 0.0, 0.0, v, EARTH.mu)
        assert el["rp"] == pytest.approx(r, rel=1e-9)      # burn at periapsis
        assert el["e"] > 0.0
        # propagate half a period -> should be at apoapsis
        s = np.array([r, 0.0, 0.0, v, 1.0])
        _, s, _ = engine.propagate(s, 0.0, kepler.period(EARTH.mu, el["a"]) / 2,
                                   1.0, EARTH)
        assert math.hypot(s[0], s[1]) == pytest.approx(el["ra"], rel=1e-4)

    def test_time_from_periapsis_full_orbit(self):
        a, e = 8.0e6, 0.1
        t_half = kepler.time_from_periapsis(a, e, math.pi, EARTH.mu)
        assert t_half == pytest.approx(kepler.period(EARTH.mu, a) / 2, rel=1e-9)


class TestThrustAndDrag:
    def test_rocket_equation_in_field_free_space(self):
        # Far from the body gravity ~ 0; a pure burn must match Tsiolkovsky.
        far = 1e12
        isp, thrust, m0, mf = 300.0, 1000.0, 100.0, 50.0
        ve = isp * 9.80665
        s = np.array([far, 0.0, 0.0, 0.0, m0])
        burn_t = (m0 - mf) * ve / thrust

        def ctrl(t, ss):
            return (thrust, 1.0, 0.0) if ss[4] > mf else (0.0, 0.0, 0.0)

        _, s, _ = engine.propagate(s, 0.0, burn_t, 0.01, EARTH,
                                   control=ctrl, ve=ve, m_min=mf)
        dv = ve * math.log(m0 / mf)
        assert s[2] == pytest.approx(dv, rel=1e-4)

    def test_drag_decays_low_orbit(self):
        # 120 km circular around Earth with CdA: energy must decrease.
        r = EARTH.radius + 120_000.0
        v = kepler.circular_velocity(EARTH.mu, r)
        s = np.array([r, 0.0, 0.0, v, 1000.0])
        e0 = 0.5 * v * v - EARTH.mu / r
        _, s, _ = engine.propagate(s, 0.0, 300.0, 0.5, EARTH, cda=10.0)
        rr = math.hypot(s[0], s[1])
        e1 = 0.5 * (s[2] ** 2 + s[3] ** 2) - EARTH.mu / rr
        assert e1 < e0
