"""Phases 11-12 + automated descent checks."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import descent_auto
import entry
import numpy as np
import pytest
import rendezvous as rz
from bodies import MOON


@pytest.fixture(scope="class")
def flight():
    return rz.fly_ascent()


class TestLunarAscent:

    def test_insertion(self, flight):
        assert flight["insertion_ok"]
        assert flight["perilune_alt"] > 15_000.0

    def test_delta_v_matches_history(self, flight):
        # Real APS insertion cost ~1,850-1,950 m/s including losses.
        assert 1_750.0 < flight["dv_ideal"] < 2_050.0

    def test_propellant_margin_exists(self, flight):
        assert flight["prop_left"] > 50.0


class TestCW:
    def setup_method(self):
        self.n = math.sqrt(MOON.mu / (MOON.radius + 111_000.0) ** 3)

    def test_two_impulse_intercept_arrives(self):
        r0, v0 = [-5_000.0, -20_000.0], [0.0, 0.0]
        T = 2.0 * math.pi / self.n
        dv1, _dv2, tot = rz.two_impulse_intercept(r0, v0, self.n, T / 2.0)
        r_end, _ = rz.cw_propagate(r0, np.array(v0) + dv1, self.n, T / 2.0)
        assert np.linalg.norm(r_end) < 1.0            # metres
        assert 1.0 < tot < 50.0                       # m/s, sane cost

    def test_lower_orbit_catches_up(self):
        """The counter-intuitive core: slow down to catch up.

        A chaser 10 km BELOW the target (x = -10 km) must drift FORWARD
        (along-track y grows) because lower orbits are faster.
        """
        r0, v0 = [-10_000.0, 0.0], [0.0, 0.0]
        # CW initial condition for a circular orbit 10 km lower is not
        # v=0; but even the v=0 drift shows the secular forward term.
        r_t, _ = rz.cw_propagate(r0, v0, self.n, 3_600.0)
        assert r_t[1] > 0.0

    def test_stm_identity_at_t0(self):
        Mrr, Mrv, _Mvr, Mvv = rz.cw_matrices(self.n, 0.0)
        assert np.allclose(Mrr, np.eye(2))
        assert np.allclose(Mvv, np.eye(2))
        assert np.allclose(Mrv, np.zeros((2, 2)))


class TestEntryCorridor:
    def test_nominal_splashes_down(self):
        r = entry.fly_entry(gamma_deg=-6.5)
        assert r["outcome"] == "splashdown"
        assert 5.0 < r["peak_g"] < 9.5
        assert r["final_v"] < 12.0            # on the mains

    def test_shallow_skips_out(self):
        r = entry.fly_entry(gamma_deg=-4.5)
        assert r["outcome"] == "skip"

    def test_steep_overloads(self):
        r = entry.fly_entry(gamma_deg=-9.0)
        assert r["outcome"] == "overload"

    def test_corridor_is_about_a_degree_wide(self):
        ok = [g / 10.0 for g in range(-75, -49, 2)
              if entry.fly_entry(gamma_deg=g / 10.0, dt=0.1)["outcome"]
              == "splashdown"]
        assert len(ok) >= 5                    # >= 1.0 deg of corridor
        assert min(ok) < -6.5 < max(ok)

    def test_ballistic_entry_is_worse(self):
        """Lift-up must reduce peak g versus no lift."""
        lift = entry.fly_entry(gamma_deg=-6.5, lift_up=True, dt=0.1)
        ball = entry.fly_entry(gamma_deg=-6.5, lift_up=False, dt=0.1)
        assert ball["peak_g"] > lift["peak_g"]


class TestAutomatedDescent:
    def test_soft_landing_under_central_gravity(self):
        r = descent_auto.fly_descent()
        assert r["landed_ok"]
        assert abs(r["v_vertical"]) <= 3.0
        assert r["v_horizontal"] <= 1.2

    def test_delta_v_matches_apollo(self):
        r = descent_auto.fly_descent()
        # Apollo 11 powered descent: ~2,100 m/s.
        assert 1_950.0 < r["dv_used"] < 2_350.0
        assert r["prop_left"] > 100.0
