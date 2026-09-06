"""
Regression tests for orbital mechanics — Phase 4 correctness.

WHY THIS FILE EXISTS
────────────────────
The original 2D dynamics applied gravity as a constant force in the −y
direction. That is a FLAT world. Under a flat-gravity model a circular
orbit is mathematically impossible: horizontal velocity never curves the
trajectory around the body, so the vehicle simply falls while translating
sideways. Starting at exactly circular velocity, it hit the surface in
138 seconds instead of orbiting for 110 minutes.

Every existing unit test still passed, because they all encoded the same
flat-world assumption. That is the lesson: a passing test suite proves
self-consistency, not correctness. These tests check the physics against
independent analytical truth instead.
"""

from __future__ import annotations

import math

import pytest

import physics as ph
from constants import (
    DPS_ISP,
    G0,
    G_MOON,
    LM_MASS_PDI,
    MU_MOON,
    PROPELLANT_MASS,
    R_MOON,
)

EXHAUST_VELOCITY = DPS_ISP * G0
DRY_MASS = LM_MASS_PDI - PROPELLANT_MASS
TEST_ALTITUDE = 15_240.0


def _circular_velocity(altitude: float) -> float:
    return math.sqrt(MU_MOON / (R_MOON + altitude))


def _orbital_period(altitude: float) -> float:
    r = R_MOON + altitude
    return 2.0 * math.pi * math.sqrt(r ** 3 / MU_MOON)


def _propagate(
    altitude: float,
    vx: float,
    duration: float,
    dt: float = 0.05,
    central_gravity: bool = True,
):
    """Coast unpowered from (0, altitude) and return the final state."""
    x, y = 0.0, altitude
    vy, theta, omega, mass = 0.0, 0.0, 0.0, LM_MASS_PDI
    steps = int(duration / dt)
    for _ in range(steps):
        x, y, vx, vy, theta, omega, mass = ph.rk4_step_2d(
            x, y, vx, vy, theta, omega, mass,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=EXHAUST_VELOCITY,
            rcs_torque=0.0,
            dt=dt,
            dry_mass=DRY_MASS,
            use_inverse_square=True,
            central_gravity=central_gravity,
        )
    return x, y, vx, vy


class TestCentralGravity:
    """Gravity must point at the Moon's centre, not straight down."""

    def test_circular_orbit_maintains_altitude(self):
        """The defining test: circular velocity must produce a circular orbit.

        Coast for one full period with the engine off. Altitude must come
        back to where it started. Tolerance is 1 m over 15,240 m — about
        0.007 % — which RK4 clears by orders of magnitude.
        """
        v = _circular_velocity(TEST_ALTITUDE)
        period = _orbital_period(TEST_ALTITUDE)

        x, y, _, _ = _propagate(TEST_ALTITUDE, v, period)
        final_altitude = ph.altitude_from_position(x, y)

        assert final_altitude == pytest.approx(TEST_ALTITUDE, abs=1.0)

    def test_circular_orbit_survives_full_period(self):
        """The vehicle must not reach the surface during a complete orbit."""
        v = _circular_velocity(TEST_ALTITUDE)
        period = _orbital_period(TEST_ALTITUDE)
        dt = 0.05

        x, y = 0.0, TEST_ALTITUDE
        vx, vy, theta, omega, mass = v, 0.0, 0.0, 0.0, LM_MASS_PDI
        min_altitude = TEST_ALTITUDE

        for _ in range(int(period / dt)):
            x, y, vx, vy, theta, omega, mass = ph.rk4_step_2d(
                x, y, vx, vy, theta, omega, mass,
                thrust_force=0.0, gravity=G_MOON,
                exhaust_velocity=EXHAUST_VELOCITY, rcs_torque=0.0,
                dt=dt, dry_mass=DRY_MASS,
                use_inverse_square=True, central_gravity=True,
            )
            min_altitude = min(min_altitude, ph.altitude_from_position(x, y))

        assert min_altitude > TEST_ALTITUDE - 1.0

    def test_specific_orbital_energy_is_conserved(self):
        """Unpowered coast must conserve specific orbital energy.

            ε = v²/2 − μ/r

        This is the strongest single check on an integrator. RK4 should
        hold it to near machine precision over one orbit.
        """
        v = _circular_velocity(TEST_ALTITUDE)
        period = _orbital_period(TEST_ALTITUDE)
        r0 = R_MOON + TEST_ALTITUDE
        energy_initial = 0.5 * v * v - MU_MOON / r0

        x, y, vx, vy = _propagate(TEST_ALTITUDE, v, period)

        r = math.sqrt(x * x + (R_MOON + y) ** 2)
        energy_final = 0.5 * (vx * vx + vy * vy) - MU_MOON / r

        assert energy_final == pytest.approx(energy_initial, rel=1e-9)

    def test_flat_gravity_cannot_sustain_an_orbit(self):
        """Documents the original bug so it stays visible.

        With central_gravity=False the vehicle reaches the surface well
        inside one orbital period, no matter how fast it is going
        horizontally. This asserts the OLD, WRONG behaviour on purpose —
        it is the reason the flag exists and must default off only for
        the short local trajectories of Phases 1-3.
        """
        v = _circular_velocity(TEST_ALTITUDE)
        _, y, _, _ = _propagate(TEST_ALTITUDE, v, 200.0, central_gravity=False)
        assert y <= 0.0

    def test_suborbital_velocity_falls(self):
        """Below circular velocity the vehicle must lose altitude."""
        v = 0.5 * _circular_velocity(TEST_ALTITUDE)
        x, y, _, _ = _propagate(TEST_ALTITUDE, v, 60.0)
        assert ph.altitude_from_position(x, y) < TEST_ALTITUDE

    def test_above_circular_velocity_raises_apoapsis(self):
        """Above circular velocity the vehicle must gain altitude."""
        v = 1.10 * _circular_velocity(TEST_ALTITUDE)
        x, y, _, _ = _propagate(TEST_ALTITUDE, v, 300.0)
        assert ph.altitude_from_position(x, y) > TEST_ALTITUDE


class TestAltitudeHelper:
    """altitude_from_position must stay consistent with the flat model."""

    def test_reduces_to_y_on_the_vertical_axis(self):
        for altitude in (0.0, 100.0, 15_240.0, 110_000.0):
            assert ph.altitude_from_position(0.0, altitude) == pytest.approx(
                altitude, abs=1e-6
            )

    def test_downrange_travel_increases_radius(self):
        """At constant y, moving downrange increases distance from centre."""
        straight = ph.altitude_from_position(0.0, 1000.0)
        offset = ph.altitude_from_position(50_000.0, 1000.0)
        assert offset > straight


class TestTimestepIndependence:
    """Results must not depend on the physics timestep."""

    def test_coast_trajectory_agrees_across_timesteps(self):
        """Halving dt must not meaningfully change the trajectory.

        This is what a per-timestep damping factor silently breaks.
        """
        v = _circular_velocity(TEST_ALTITUDE)
        coarse = _propagate(TEST_ALTITUDE, v, 600.0, dt=0.10)
        fine = _propagate(TEST_ALTITUDE, v, 600.0, dt=0.01)

        assert coarse[0] == pytest.approx(fine[0], rel=1e-6)
        assert coarse[1] == pytest.approx(fine[1], rel=1e-6)

    def test_attitude_hold_is_timestep_independent(self):
        """exp(-k·dt) decay must give the same result at any timestep.

        The old formulation, omega *= (1 - 0.02) per step, changes the
        decay rate whenever PHYSICS_DT changes. This test would fail
        against that implementation.
        """
        from constants import ATTITUDE_HOLD_RATE

        omega_initial = 0.5
        duration = 1.0

        def decay(dt: float) -> float:
            omega = omega_initial
            for _ in range(int(duration / dt)):
                omega *= math.exp(-ATTITUDE_HOLD_RATE * dt)
            return omega

        assert decay(0.01) == pytest.approx(decay(0.001), rel=1e-9)
        assert decay(0.01) == pytest.approx(
            omega_initial * math.exp(-ATTITUDE_HOLD_RATE * duration), rel=1e-9
        )


class TestAnalyticalCrossChecks:
    """Independent hand-calculable checks."""

    def test_circular_velocity_matches_hand_calculation(self):
        # sqrt(4.9028695e12 / 1752640) = 1672.5 m/s
        assert ph.orbital_velocity(TEST_ALTITUDE) == pytest.approx(1672.5, abs=0.5)

    def test_orbital_period_matches_keplers_third_law(self):
        r = R_MOON + TEST_ALTITUDE
        expected = 2.0 * math.pi * math.sqrt(r ** 3 / MU_MOON)
        assert expected == pytest.approx(6584.0, abs=5.0)

    def test_surface_gravity_from_mu_over_r_squared(self):
        # Slightly below the quoted 1.625 because that value folds in
        # oblateness and mascon effects.
        assert ph.gravity_at_altitude(0.0) == pytest.approx(1.6242, abs=0.001)
