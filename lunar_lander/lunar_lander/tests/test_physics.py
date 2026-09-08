"""
Tests for the Apollo LM descent simulator physics engine.

Phase 1 tests: Free-fall, thrust acceleration, energy conservation, landing.
Phase 2 tests: Rocket equation, mass flow, fuel exhaustion, TWR scaling, RK4 mass coupling.
Phase 3 tests: 2D thrust decomposition, RCS angular dynamics, multi-criteria landing check.

All tests use hand-computed reference values from the real Apollo 11 LM parameters.
"""

from __future__ import annotations

import math

# Import physics functions
from physics import (
    rk4_step,
    rk4_step_with_mass,
    check_landing,
    compute_delta_v_remaining,
    compute_mass_flow_rate,
    derivatives_2d,
    rk4_step_2d,
    check_landing_2d,
    compute_moment_of_inertia,
    # Phase 4
    gravity_at_altitude,
    orbital_velocity,
    vis_viva_velocity,
    compute_orbital_elements,
)

# Import constants
from constants import (
    G_MOON,
    G0,
    DPS_MAX_THRUST,
    DPS_ISP,
    LM_MASS_PDI,
    PROPELLANT_MASS,
    MAX_VERTICAL_SPEED,
    RCS_ATTITUDE_TORQUE,
    LM_MOMENT_OF_INERTIA_FULL,
    LM_MOMENT_OF_INERTIA_EMPTY,
    MU_MOON,
    R_MOON,
    DESCENT_ORBIT_PERILUNE,
    DESCENT_ORBIT_APOLUNE,
)


# ============================================================================
# PHASE 1 TESTS — CONSTANT-MASS DYNAMICS
# ============================================================================

class TestFreeFall:
    """Verify free-fall against the analytical solution.

    For constant gravity with no thrust:
        h(t) = h₀ + v₀t − ½gt²
        v(t) = v₀ − gt

    With RK4 and constant acceleration, the result should be exact
    (to machine precision) because RK4 is exact for polynomials up
    to degree 4, and the solution is quadratic.
    """

    def test_freefall_10_seconds(self) -> None:
        """Drop from 2000m at rest for 10 seconds."""
        h, v = 2000.0, 0.0
        dt = 0.01
        for _ in range(1000):  # 10 seconds at 100 Hz
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)

        t = 10.0
        expected_h = 2000.0 - 0.5 * G_MOON * t**2
        expected_v = -G_MOON * t

        assert abs(h - expected_h) < 1e-6, f"h={h}, expected={expected_h}"
        assert abs(v - expected_v) < 1e-6, f"v={v}, expected={expected_v}"

    def test_freefall_with_initial_velocity(self) -> None:
        """Drop from 2000m at -50 m/s for 10 seconds."""
        h, v = 2000.0, -50.0
        dt = 0.01
        for _ in range(1000):
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)

        t = 10.0
        expected_h = 2000.0 + (-50.0) * t - 0.5 * G_MOON * t**2
        expected_v = -50.0 - G_MOON * t

        assert abs(h - expected_h) < 1e-6
        assert abs(v - expected_v) < 1e-6

    def test_freefall_velocity_linearity(self) -> None:
        """Velocity under constant gravity should increase linearly."""
        h, v = 10000.0, 0.0
        dt = 0.01
        velocities = [v]

        for _ in range(500):  # 5 seconds
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)
            velocities.append(v)

        for i in range(1, len(velocities)):
            expected = -G_MOON * i * dt
            assert abs(velocities[i] - expected) < 1e-6


class TestThrustAcceleration:
    """Verify thrust acceleration calculations."""

    def test_thrust_acceleration_magnitude(self) -> None:
        """Full thrust acceleration = F/m = 45040/15100 ≈ 2.9828 m/s²."""
        thrust_accel = DPS_MAX_THRUST / LM_MASS_PDI
        expected = 45040.0 / 15100.0
        assert abs(thrust_accel - expected) < 1e-4

    def test_full_thrust_from_rest_1_second(self) -> None:
        """1 second of full thrust from rest, checking velocity."""
        h, v = 1000.0, 0.0
        dt = 0.01
        thrust_accel = DPS_MAX_THRUST / LM_MASS_PDI
        net_accel = thrust_accel - G_MOON

        for _ in range(100):
            h, v = rk4_step(h, v, thrust_accel=thrust_accel, gravity=G_MOON, dt=dt)

        expected_v = net_accel * 1.0
        assert abs(v - expected_v) < 1e-4, f"v={v}, expected={expected_v}"

    def test_hover_thrust(self) -> None:
        """Thrust exactly equal to gravity should keep velocity at zero."""
        h, v = 1000.0, 0.0
        dt = 0.01
        thrust_accel = G_MOON  # Exactly cancels gravity

        for _ in range(10000):  # 100 seconds
            h, v = rk4_step(h, v, thrust_accel=thrust_accel, gravity=G_MOON, dt=dt)

        assert abs(v) < 1e-6, f"Velocity should be ~0, got {v}"
        assert abs(h - 1000.0) < 1e-4, f"Altitude should be ~1000, got {h}"


class TestEnergyConservation:
    """Verify energy conservation in free-fall.

    Total mechanical energy E = ½mv² + mgh should be conserved
    when there is no thrust (no external energy input).
    """

    @staticmethod
    def energy(h: float, v: float, m: float = 1.0, g: float = G_MOON) -> float:
        return 0.5 * m * v**2 + m * g * h

    def test_energy_conservation_100_steps(self) -> None:
        h, v = 5000.0, 0.0
        E0 = self.energy(h, v)
        dt = 0.01

        for _ in range(100):
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)

        E_final = self.energy(h, v)
        assert abs(E_final - E0) / abs(E0) < 1e-10

    def test_energy_conservation_long_coast(self) -> None:
        h, v = 50000.0, -100.0
        E0 = self.energy(h, v)
        dt = 0.01

        for _ in range(10000):
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)

        E_final = self.energy(h, v)
        assert abs(E_final - E0) / abs(E0) < 1e-10

    def test_energy_partition(self) -> None:
        h, v = 2000.0, 0.0
        dt = 0.01

        for _ in range(500):
            h, v = rk4_step(h, v, thrust_accel=0.0, gravity=G_MOON, dt=dt)

        KE = 0.5 * v**2
        PE = G_MOON * h
        KE0 = 0.0
        PE0 = G_MOON * 2000.0

        assert abs((KE + PE) - (KE0 + PE0)) < 1e-6


class TestLandingDetection:
    """Verify the landing detection logic."""

    def test_flying_at_altitude(self) -> None:
        assert check_landing(100.0, -5.0) == "flying"

    def test_soft_landing_slow(self) -> None:
        assert check_landing(0.0, -1.0) == "landed"

    def test_soft_landing_at_limit(self) -> None:
        assert check_landing(0.0, -MAX_VERTICAL_SPEED) == "landed"

    def test_crash_exceeds_limit(self) -> None:
        assert check_landing(0.0, -(MAX_VERTICAL_SPEED + 0.001)) == "crashed"

    def test_crash_below_surface(self) -> None:
        assert check_landing(-5.0, -50.0) == "crashed"

    def test_boundary_precision(self) -> None:
        assert check_landing(0.001, -100.0) == "flying"


# ============================================================================
# PHASE 2 TESTS — VARIABLE-MASS DYNAMICS
# ============================================================================

class TestRocketEquation:
    """Verify the Tsiolkovsky rocket equation implementation.

    Δv = v_e × ln(m₀ / m_f) = Isp × g0 × ln(m₀ / m_f)

    At PDI:
        Δv = 311 × 9.80665 × ln(15100 / 6900)
           = 3049.87 × ln(2.18841...)
           = 3049.87 × 0.78315...
           ≈ 2388.5 m/s
    """

    def test_delta_v_at_pdi(self) -> None:
        """Full propellant load should give ~2388 m/s of delta-v."""
        dv = compute_delta_v_remaining(
            mass_current=LM_MASS_PDI,
            mass_dry=LM_MASS_PDI - PROPELLANT_MASS,
        )
        # Hand calculation:
        v_e = DPS_ISP * G0  # 311 × 9.80665 = 3049.87
        expected = v_e * math.log(15100.0 / 6900.0)
        assert abs(dv - expected) < 0.1, f"Δv={dv}, expected={expected}"
        # Sanity: should be near 2388 m/s
        assert 2380 < dv < 2400, f"Δv={dv} not in expected range"

    def test_delta_v_half_fuel(self) -> None:
        """Half propellant remaining gives MORE than half delta-v.

        This is counter-intuitive but correct:
        The LAST half of propellant provides more Δv than the FIRST half
        because the vehicle is lighter during the second half of the burn.

        Full:  Δv = v_e × ln(15100/6900) = 3050 × 0.783 ≈ 2389 m/s
        Half:  Δv = v_e × ln(11000/6900) = 3050 × 0.466 ≈ 1422 m/s

        The first 4,100 kg gave: 2389 - 1422 = 967 m/s
        The last  4,100 kg gives: 1422 m/s
        That's 47% more from the same mass — because it was lighter.

        This is why the rocket equation is "tyrannical": the first
        propellant you burn is the most expensive because it has to
        lift ALL the rest of the propellant too.
        """
        dv_full = compute_delta_v_remaining(LM_MASS_PDI, LM_MASS_PDI - PROPELLANT_MASS)
        half_prop = PROPELLANT_MASS / 2
        dv_half = compute_delta_v_remaining(
            LM_MASS_PDI - half_prop,
            LM_MASS_PDI - PROPELLANT_MASS,
        )
        # The remaining half of fuel gives MORE than half the total Δv
        assert dv_half > dv_full / 2, (
            f"Half fuel Δv ({dv_half:.1f}) should be > half of full Δv ({dv_full/2:.1f})"
        )
        # The first half gave LESS than the second half
        dv_first_half = dv_full - dv_half
        assert dv_first_half < dv_half, (
            f"First half Δv ({dv_first_half:.1f}) should be < second half ({dv_half:.1f})"
        )

    def test_delta_v_no_fuel(self) -> None:
        """Zero propellant should give zero delta-v."""
        dv = compute_delta_v_remaining(6900.0, 6900.0)
        assert dv == 0.0

    def test_delta_v_negative_propellant(self) -> None:
        """Mass below dry mass should return zero (guard case)."""
        dv = compute_delta_v_remaining(5000.0, 6900.0)
        assert dv == 0.0


class TestMassFlowRate:
    """Verify propellant consumption rate calculations.

    ṁ = F / (Isp × g0) = F / v_e

    At full thrust:  ṁ = 45,040 / (311 × 9.80665) ≈ 14.77 kg/s
    At 10% throttle: ṁ = 4,504 / (311 × 9.80665) ≈  1.477 kg/s
    """

    def test_full_thrust_flow_rate(self) -> None:
        """Full thrust mass flow should be ~14.77 kg/s."""
        flow = compute_mass_flow_rate(DPS_MAX_THRUST)
        expected = DPS_MAX_THRUST / (DPS_ISP * G0)
        assert abs(flow - expected) < 0.01
        assert 14.7 < flow < 14.9, f"Flow rate {flow} not in expected range"

    def test_partial_throttle_flow(self) -> None:
        """Mass flow at 30% throttle should be 30% of full."""
        flow_full = compute_mass_flow_rate(DPS_MAX_THRUST)
        flow_30 = compute_mass_flow_rate(0.30 * DPS_MAX_THRUST)
        assert abs(flow_30 - 0.30 * flow_full) < 0.001

    def test_zero_thrust_flow(self) -> None:
        """No thrust should mean no propellant consumption."""
        assert compute_mass_flow_rate(0.0) == 0.0

    def test_one_second_mass_decrease(self) -> None:
        """Integrate mass depletion for 1 second at full thrust.

        After 1 second: mass should decrease by ~14.77 kg.
        """
        v_e = DPS_ISP * G0
        mass = LM_MASS_PDI
        _, _, new_mass = rk4_step_with_mass(
            altitude=2000.0,
            velocity=0.0,
            mass=mass,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=v_e,
            dt=1.0,
            dry_mass=LM_MASS_PDI - PROPELLANT_MASS,
        )
        dm = mass - new_mass
        expected_dm = DPS_MAX_THRUST / v_e  # ≈ 14.77 kg
        assert abs(dm - expected_dm) < 0.1, f"Δm={dm}, expected={expected_dm}"


class TestVariableMassRK4:
    """Verify the mass-coupled RK4 integrator.

    The key test: RK4 with mass coupling should produce different (more
    accurate) results than naively updating mass after the step.
    """

    def test_free_fall_mass_unchanged(self) -> None:
        """With no thrust, mass should not change."""
        v_e = DPS_ISP * G0
        new_h, new_v, new_m = rk4_step_with_mass(
            altitude=2000.0,
            velocity=-50.0,
            mass=15100.0,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=v_e,
            dt=0.01,
        )
        assert new_m == 15100.0, "Mass should not change without thrust"

    def test_thrust_acceleration_increases_with_decreasing_mass(self) -> None:
        """TWR increases as the vehicle empties.

        At full mass:  a = 45040/15100 = 2.983 m/s²
        At dry mass:   a = 45040/6900  = 6.527 m/s²

        This 2.2× increase is the core Phase 2 learning.
        """
        a_full = DPS_MAX_THRUST / LM_MASS_PDI
        a_empty = DPS_MAX_THRUST / (LM_MASS_PDI - PROPELLANT_MASS)
        ratio = a_empty / a_full
        assert 2.1 < ratio < 2.3, f"Acceleration ratio = {ratio}, expected ~2.19"

    def test_rk4_mass_coupling_vs_naive(self) -> None:
        """RK4 with mass inside the step should differ from naive update.

        Naive: compute step at constant mass, then subtract ṁ×dt.
        Correct: mass changes within the RK4 sub-steps.

        The difference is tiny for dt=0.01 but measurable for dt=1.0.
        """
        v_e = DPS_ISP * G0
        dry = LM_MASS_PDI - PROPELLANT_MASS
        dt = 1.0  # Large step to make the difference visible

        # Correct: mass-coupled RK4
        _, v_rk4, m_rk4 = rk4_step_with_mass(
            altitude=2000.0, velocity=0.0, mass=LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST, gravity=G_MOON,
            exhaust_velocity=v_e, dt=dt, dry_mass=dry,
        )

        # Naive: constant-mass step + post-hoc mass update
        thrust_accel = DPS_MAX_THRUST / LM_MASS_PDI
        _, v_naive = rk4_step(
            altitude=2000.0, velocity=0.0,
            thrust_accel=thrust_accel, gravity=G_MOON, dt=dt,
        )

        # The velocities should differ because the correct version
        # uses decreasing mass (increasing acceleration) during the step.
        # With decreasing mass, acceleration is HIGHER on average,
        # so v_rk4 should be slightly larger (more positive) than v_naive.
        diff = abs(v_rk4 - v_naive)
        assert diff > 0.001, (
            f"RK4-coupled vs naive difference too small: {diff:.6f} m/s. "
            f"Mass coupling should produce a measurable effect at dt=1.0."
        )

    def test_fuel_exhaustion_clamp(self) -> None:
        """Mass should never go below dry mass."""
        v_e = DPS_ISP * G0
        dry = LM_MASS_PDI - PROPELLANT_MASS

        # Start with almost no fuel
        mass = dry + 1.0  # Only 1 kg of propellant

        _, _, new_mass = rk4_step_with_mass(
            altitude=2000.0, velocity=0.0,
            mass=mass, thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON, exhaust_velocity=v_e,
            dt=0.01, dry_mass=dry,
        )
        assert new_mass >= dry, f"Mass {new_mass} went below dry mass {dry}"

    def test_long_burn_mass_depletion(self) -> None:
        """Run a full-power burn for 100 seconds, check mass consistency.

        At ṁ ≈ 14.77 kg/s, 100 seconds should consume ~1477 kg.
        Starting mass: 15,100. Expected mass: ~13,623 kg.
        """
        v_e = DPS_ISP * G0
        dry = LM_MASS_PDI - PROPELLANT_MASS
        h, v, m = 50000.0, 0.0, float(LM_MASS_PDI)
        dt = 0.01
        steps = 10000  # 100 seconds

        for _ in range(steps):
            h, v, m = rk4_step_with_mass(
                altitude=h, velocity=v, mass=m,
                thrust_force=DPS_MAX_THRUST, gravity=G_MOON,
                exhaust_velocity=v_e, dt=dt, dry_mass=dry,
            )

        expected_dm = DPS_MAX_THRUST / v_e * 100.0  # ~1477 kg
        actual_dm = LM_MASS_PDI - m
        # Allow ~1% tolerance for RK4 integration accuracy over 100s
        assert abs(actual_dm - expected_dm) / expected_dm < 0.01, (
            f"Mass depleted: {actual_dm:.1f} kg, expected ~{expected_dm:.1f} kg"
        )

        # Mass should still be well above dry mass
        assert m > dry + 5000, f"Mass {m} should be ~13623 kg"


class TestVehicleModel:
    """Test the LunarModule vehicle model (Phase 2)."""

    def test_throttle_bucket(self) -> None:
        """Throttle should skip 60-100% forbidden zone."""
        from vehicle import LunarModule

        lm = LunarModule()

        # Ramp up from off to max continuous
        lm.throttle_up()  # → 10%
        assert abs(lm.throttle - 0.10) < 0.001

        for _ in range(10):  # → 15% → 20% → ... → 60%
            lm.throttle_up()

        assert abs(lm.throttle - 0.60) < 0.001, f"Should be at 60%, got {lm.throttle}"

        # Next up should jump to 100% (skip the bucket)
        lm.throttle_up()
        assert abs(lm.throttle - 1.00) < 0.001, f"Should jump to 100%, got {lm.throttle}"

        # Down from 100% should drop to 60%
        lm.throttle_down()
        assert abs(lm.throttle - 0.60) < 0.001, f"Should drop to 60%, got {lm.throttle}"

    def test_throttle_off(self) -> None:
        """Throttle down from minimum should turn off."""
        from vehicle import LunarModule

        lm = LunarModule()
        lm.throttle_up()  # → 10%
        assert lm.engine_on

        lm.throttle_down()  # → off
        assert abs(lm.throttle) < 0.001
        assert not lm.engine_on

    def test_mass_decreases_with_burn(self) -> None:
        """Vehicle mass should decrease as propellant burns."""
        from vehicle import LunarModule

        lm = LunarModule()
        initial_mass = lm.mass
        initial_prop = lm.propellant

        # Simulate physics deducting 100 kg of mass
        lm.sync_mass_from_physics(initial_mass - 100.0)

        assert abs(lm.mass - (initial_mass - 100.0)) < 0.001
        assert abs(lm.propellant - (initial_prop - 100.0)) < 0.001
        assert not lm.fuel_exhausted

    def test_fuel_exhaustion(self) -> None:
        """Syncing mass at dry mass should trigger fuel_exhausted."""
        from vehicle import LunarModule

        lm = LunarModule()
        lm.throttle_up()  # Engine on at 10%
        assert lm.engine_on

        # Deplete all propellant
        lm.sync_mass_from_physics(lm.dry_mass)

        assert lm.fuel_exhausted
        assert abs(lm.propellant) < 0.001
        assert not lm.engine_on
        assert abs(lm.throttle) < 0.001
        assert lm.thrust_force == 0.0

    def test_delta_v_remaining(self) -> None:
        """Vehicle delta-v should match hand calculation."""
        from vehicle import LunarModule

        lm = LunarModule()
        dv = lm.delta_v_remaining
        expected = DPS_ISP * G0 * math.log(LM_MASS_PDI / (LM_MASS_PDI - PROPELLANT_MASS))
        assert abs(dv - expected) < 0.1

    def test_twr_at_full_and_empty(self) -> None:
        """TWR should increase from ~1.84 to ~4.02 as fuel burns."""
        from vehicle import LunarModule

        lm = LunarModule()
        lm.throttle = 1.0  # Full thrust

        twr_full = lm.twr
        assert 1.8 < twr_full < 1.9, f"TWR at full mass: {twr_full}"

        # Drain all fuel
        lm.sync_mass_from_physics(lm.dry_mass + 1.0)  # 1 kg left
        lm.throttle = 1.0  # Re-enable (sync turned it off for 0 fuel)
        lm.fuel_exhausted = False  # Override for test

        twr_empty = lm.twr
        assert 3.9 < twr_empty < 4.1, f"TWR at empty: {twr_empty}"


# ============================================================================
# PHASE 3 TESTS — 2D DYNAMICS, RCS, MULTI-CRITERIA LANDING
# ============================================================================

class TestThrustDecomposition:
    """Verify that thrust decomposes correctly by vehicle angle.

    When the LM is tilted by angle θ from vertical:
        F_horizontal = F × sin(θ)
        F_vertical   = F × cos(θ)

    At θ = 0: all thrust is vertical (Phase 2 case).
    At θ = 30°: cos(30°)=0.866, sin(30°)=0.500.
    At θ = 90°: cos(90°)=0, sin(90°)=1 — all thrust horizontal.
    """

    VE = DPS_ISP * G0  # Exhaust velocity

    def test_upright_matches_1d(self) -> None:
        """At θ = 0, the 2D equations must match Phase 2 (pure vertical)."""
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=-50,
            theta=0.0, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        # d[2] = dvx = F/m * sin(0) = 0
        assert abs(d[2]) < 1e-10, "Horizontal thrust should be zero when upright"
        # d[3] = dvy = F/m * cos(0) - g = F/m - g
        expected_vy = DPS_MAX_THRUST / LM_MASS_PDI - G_MOON
        assert abs(d[3] - expected_vy) < 1e-6

    def test_30_degrees(self) -> None:
        """At θ = 30° (π/6), verify thrust decomposition.

        cos(30°) = √3/2 ≈ 0.8660
        sin(30°) = 1/2  = 0.5000

        F/m = 45040 / 15100 = 2.9828 m/s²
        dvx = 2.9828 × 0.5     = 1.4914 m/s²
        dvy = 2.9828 × 0.866 − 1.625 = 0.9580 m/s²
        """
        theta = math.radians(30.0)
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=theta, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        thrust_accel = DPS_MAX_THRUST / LM_MASS_PDI
        expected_dvx = thrust_accel * math.sin(theta)
        expected_dvy = thrust_accel * math.cos(theta) - G_MOON

        assert abs(d[2] - expected_dvx) < 0.01, \
            f"dvx at 30°: {d[2]:.4f} vs {expected_dvx:.4f}"
        assert abs(d[3] - expected_dvy) < 0.01, \
            f"dvy at 30°: {d[3]:.4f} vs {expected_dvy:.4f}"

    def test_90_degrees_no_vertical_thrust(self) -> None:
        """At θ = 90°, cos(90°) ≈ 0 → no vertical thrust, only gravity."""
        theta = math.radians(90.0)
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=theta, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        # dvy ≈ F/m × cos(90°) - g ≈ 0 - 1.625
        assert abs(d[3] - (-G_MOON)) < 0.01, \
            "At 90° tilt, vertical thrust should be negligible"
        # dvx ≈ F/m × sin(90°) = F/m
        expected_dvx = DPS_MAX_THRUST / LM_MASS_PDI
        assert abs(d[2] - expected_dvx) < 0.01

    def test_negative_angle(self) -> None:
        """Negative θ (tilted left) should give negative horizontal thrust."""
        theta = math.radians(-30.0)
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=theta, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        assert d[2] < 0, "Negative tilt should produce negative horizontal thrust"

    def test_no_thrust_is_freefall(self) -> None:
        """With no thrust, only gravity acts (regardless of angle)."""
        theta = math.radians(45.0)
        d = derivatives_2d(
            x=0, y=1000, vx=5.0, vy=-10.0,
            theta=theta, omega=0.5, mass=LM_MASS_PDI,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        assert abs(d[2]) < 1e-10, "No horizontal accel without thrust"
        assert abs(d[3] - (-G_MOON)) < 1e-10, "Only gravity vertically"
        assert abs(d[6]) < 1e-10, "No mass change without thrust"


class TestRCSAngularDynamics:
    """Verify RCS torque produces correct angular acceleration.

    α = τ / I

    At full mass:  α = 890 / 28000 ≈ 0.0318 rad/s²  ≈ 1.82 °/s²
    At dry mass:   α = 890 / 12000 ≈ 0.0742 rad/s²  ≈ 4.25 °/s²
    """

    VE = DPS_ISP * G0

    def test_angular_accel_full_mass(self) -> None:
        """RCS angular acceleration at full PDI mass."""
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=0.0, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=RCS_ATTITUDE_TORQUE,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        expected_alpha = RCS_ATTITUDE_TORQUE / LM_MOMENT_OF_INERTIA_FULL
        assert abs(d[5] - expected_alpha) < 1e-6, \
            f"α at full mass: {d[5]:.6f} vs {expected_alpha:.6f}"

    def test_angular_accel_empty_mass(self) -> None:
        """RCS angular acceleration at dry mass — should be ~2.3× higher."""
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=0.0, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=RCS_ATTITUDE_TORQUE,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_EMPTY,
        )
        expected_alpha = RCS_ATTITUDE_TORQUE / LM_MOMENT_OF_INERTIA_EMPTY
        assert abs(d[5] - expected_alpha) < 1e-6

    def test_negative_torque(self) -> None:
        """Negative torque should produce negative angular acceleration."""
        d = derivatives_2d(
            x=0, y=1000, vx=0, vy=0,
            theta=0.0, omega=0.0, mass=LM_MASS_PDI,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=-RCS_ATTITUDE_TORQUE,
            moment_of_inertia=LM_MOMENT_OF_INERTIA_FULL,
        )
        assert d[5] < 0, "Negative torque → negative angular acceleration"

    def test_rcs_rotation_over_time(self) -> None:
        """Apply RCS for 3 seconds: ω should reach ~0.095 rad/s ≈ 5.5 °/s."""
        x, y, vx, vy = 0.0, 1000.0, 0.0, 0.0
        theta, omega = 0.0, 0.0
        mass = LM_MASS_PDI
        dt = 0.01
        ve = self.VE
        dry = LM_MASS_PDI - PROPELLANT_MASS

        for _ in range(300):  # 3 seconds
            x, y, vx, vy, theta, omega, mass = rk4_step_2d(
                x, y, vx, vy, theta, omega, mass,
                thrust_force=0.0,
                gravity=G_MOON,
                exhaust_velocity=ve,
                rcs_torque=RCS_ATTITUDE_TORQUE,
                dt=dt,
                dry_mass=dry,
            )

        # Expected: ω ≈ α × t = (890/28000) × 3 ≈ 0.0954 rad/s
        expected_omega = (RCS_ATTITUDE_TORQUE / LM_MOMENT_OF_INERTIA_FULL) * 3.0
        assert abs(omega - expected_omega) < 0.005, \
            f"ω after 3s: {omega:.4f} vs {expected_omega:.4f}"

        # Theta should be ≈ ½αt² = 0.5 × 0.0318 × 9 = 0.143 rad ≈ 8.2°
        expected_theta = 0.5 * (RCS_ATTITUDE_TORQUE / LM_MOMENT_OF_INERTIA_FULL) * 9.0
        assert abs(theta - expected_theta) < 0.01, \
            f"θ after 3s: {math.degrees(theta):.1f}° vs {math.degrees(expected_theta):.1f}°"


class TestMomentOfInertia:
    """Verify MOI interpolation between full and empty."""

    def test_full_mass(self) -> None:
        moi = compute_moment_of_inertia(LM_MASS_PDI)
        assert abs(moi - LM_MOMENT_OF_INERTIA_FULL) < 1.0

    def test_dry_mass(self) -> None:
        dry = LM_MASS_PDI - PROPELLANT_MASS
        moi = compute_moment_of_inertia(dry)
        assert abs(moi - LM_MOMENT_OF_INERTIA_EMPTY) < 1.0

    def test_half_mass(self) -> None:
        half = LM_MASS_PDI - PROPELLANT_MASS / 2.0
        moi = compute_moment_of_inertia(half)
        expected = (LM_MOMENT_OF_INERTIA_FULL + LM_MOMENT_OF_INERTIA_EMPTY) / 2.0
        assert abs(moi - expected) < 1.0

    def test_monotonically_decreases(self) -> None:
        """MOI should decrease as mass decreases."""
        dry = LM_MASS_PDI - PROPELLANT_MASS
        masses = [LM_MASS_PDI, LM_MASS_PDI - 2000, LM_MASS_PDI - 4000, dry]
        mois = [compute_moment_of_inertia(m) for m in masses]
        for i in range(len(mois) - 1):
            assert mois[i] > mois[i + 1], \
                f"MOI should decrease: {mois[i]} > {mois[i + 1]}"


class TestFreeFall2D:
    """Verify 2D free-fall conserves horizontal velocity and matches 1D vertical."""

    VE = DPS_ISP * G0

    def test_horizontal_velocity_constant(self) -> None:
        """With no thrust and no drag, horizontal velocity must be constant."""
        x, y, vx, vy = 0.0, 2000.0, 50.0, 0.0
        theta, omega = 0.0, 0.0
        mass = LM_MASS_PDI
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        for _ in range(1000):  # 10 seconds
            x, y, vx, vy, theta, omega, mass = rk4_step_2d(
                x, y, vx, vy, theta, omega, mass,
                thrust_force=0.0,
                gravity=G_MOON,
                exhaust_velocity=self.VE,
                rcs_torque=0.0,
                dt=dt,
                dry_mass=dry,
            )

        assert abs(vx - 50.0) < 1e-6, f"vx drifted: {vx}"
        # x should be ≈ 50 × 10 = 500 m
        assert abs(x - 500.0) < 0.01

    def test_vertical_matches_1d(self) -> None:
        """Vertical free-fall in 2D should match the analytical solution."""
        x, y, vx, vy = 0.0, 2000.0, 0.0, 0.0
        theta, omega = 0.0, 0.0
        mass = LM_MASS_PDI
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        for _ in range(1000):  # 10 seconds
            x, y, vx, vy, theta, omega, mass = rk4_step_2d(
                x, y, vx, vy, theta, omega, mass,
                thrust_force=0.0,
                gravity=G_MOON,
                exhaust_velocity=self.VE,
                rcs_torque=0.0,
                dt=dt,
                dry_mass=dry,
            )

        # Analytical: y = 2000 - 0.5 × 1.625 × 100 = 2000 - 81.25 = 1918.75
        expected_y = 2000.0 - 0.5 * G_MOON * 100.0
        assert abs(y - expected_y) < 0.01, f"y: {y:.4f} vs {expected_y:.4f}"

        # Analytical: vy = -1.625 × 10 = -16.25
        expected_vy = -G_MOON * 10.0
        assert abs(vy - expected_vy) < 0.01


class TestLanding2D:
    """Test all landing criteria combinations.

    There are four failure modes:
        1. Too fast vertically (> 3.0 m/s)
        2. Too fast horizontally (> 1.2 m/s)
        3. Too tilted (> 12°)
        4. No propellant
    """

    def test_perfect_landing(self) -> None:
        status, violations = check_landing_2d(
            y=0.0, vx=0.0, vy=-0.5, theta=0.0, propellant=100.0,
        )
        assert status == "landed"
        assert violations == []

    def test_still_flying(self) -> None:
        status, violations = check_landing_2d(
            y=100.0, vx=50.0, vy=-100.0, theta=1.0, propellant=0.0,
        )
        assert status == "flying"
        assert violations == []

    def test_vertical_too_fast(self) -> None:
        status, violations = check_landing_2d(
            y=0.0, vx=0.0, vy=-5.0, theta=0.0, propellant=100.0,
        )
        assert status == "crashed"
        assert any("Vertical" in v for v in violations)

    def test_horizontal_too_fast(self) -> None:
        status, violations = check_landing_2d(
            y=0.0, vx=3.0, vy=-0.5, theta=0.0, propellant=100.0,
        )
        assert status == "crashed"
        assert any("Horizontal" in v for v in violations)

    def test_too_tilted(self) -> None:
        theta = math.radians(15.0)  # > 12° limit
        status, violations = check_landing_2d(
            y=0.0, vx=0.0, vy=-0.5, theta=theta, propellant=100.0,
        )
        assert status == "crashed"
        assert any("Tilt" in v for v in violations)

    def test_no_fuel(self) -> None:
        status, violations = check_landing_2d(
            y=0.0, vx=0.0, vy=-0.5, theta=0.0, propellant=0.0,
        )
        assert status == "crashed"
        assert any("propellant" in v.lower() for v in violations)

    def test_multiple_violations(self) -> None:
        """All limits exceeded simultaneously → should list all violations."""
        theta = math.radians(20.0)
        status, violations = check_landing_2d(
            y=0.0, vx=5.0, vy=-10.0, theta=theta, propellant=0.0,
        )
        assert status == "crashed"
        assert len(violations) == 4, f"Expected 4 violations, got {len(violations)}"

    def test_just_within_limits(self) -> None:
        """Landing exactly at every limit should succeed."""
        theta = math.radians(11.9)  # < 12° limit
        status, violations = check_landing_2d(
            y=0.0, vx=1.1, vy=-2.9, theta=theta, propellant=0.1,
        )
        assert status == "landed"
        assert violations == []

    def test_negative_theta_same_as_positive(self) -> None:
        """Tilt limit applies to absolute value."""
        theta_pos = math.radians(15.0)
        theta_neg = math.radians(-15.0)

        _, viol_pos = check_landing_2d(
            y=0.0, vx=0.0, vy=-0.5, theta=theta_pos, propellant=100.0,
        )
        _, viol_neg = check_landing_2d(
            y=0.0, vx=0.0, vy=-0.5, theta=theta_neg, propellant=100.0,
        )
        assert len(viol_pos) == len(viol_neg) == 1


class TestRK4_2D_MassCoupling:
    """Verify mass coupling works correctly in the 2D integrator."""

    VE = DPS_ISP * G0

    def test_mass_decreases_under_thrust(self) -> None:
        """Mass should decrease when engine fires."""
        x, y, vx, vy = 0.0, 2000.0, 0.0, 0.0
        theta, omega = 0.0, 0.0
        mass = LM_MASS_PDI
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        _, _, _, _, _, _, new_mass = rk4_step_2d(
            x, y, vx, vy, theta, omega, mass,
            thrust_force=DPS_MAX_THRUST,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            dt=dt,
            dry_mass=dry,
        )
        assert new_mass < mass, "Mass should decrease under thrust"
        # Expected: dm ≈ F/ve × dt = 45040/3050 × 0.01 ≈ 0.148 kg
        expected_dm = DPS_MAX_THRUST / self.VE * dt
        actual_dm = mass - new_mass
        assert abs(actual_dm - expected_dm) < 0.01

    def test_mass_constant_without_thrust(self) -> None:
        """Mass should not change in free-fall."""
        x, y, vx, vy = 0.0, 2000.0, 10.0, -5.0
        theta, omega = 0.1, 0.0
        mass = LM_MASS_PDI
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        _, _, _, _, _, _, new_mass = rk4_step_2d(
            x, y, vx, vy, theta, omega, mass,
            thrust_force=0.0,
            gravity=G_MOON,
            exhaust_velocity=self.VE,
            rcs_torque=0.0,
            dt=dt,
            dry_mass=dry,
        )
        assert abs(new_mass - mass) < 1e-10


class TestVehicleRCS:
    """Test RCS control methods on the LunarModule."""

    def test_rcs_commands(self) -> None:
        from vehicle import LunarModule
        lm = LunarModule()

        assert lm.rcs_command == 0
        lm.rcs_left()
        assert lm.rcs_command == -1
        lm.rcs_right()
        assert lm.rcs_command == 1
        lm.rcs_stop()
        assert lm.rcs_command == 0

    def test_rcs_torque_values(self) -> None:
        from vehicle import LunarModule
        lm = LunarModule()

        lm.rcs_left()
        assert lm.rcs_torque_value == -RCS_ATTITUDE_TORQUE
        lm.rcs_right()
        assert lm.rcs_torque_value == RCS_ATTITUDE_TORQUE
        lm.rcs_stop()
        assert lm.rcs_torque_value == 0.0


# ============================================================================
# PHASE 4 TESTS — INVERSE-SQUARE GRAVITY, ORBITAL MECHANICS, GUIDANCE
# ============================================================================

class TestInverseSquareGravity:
    """Verify inverse-square gravity against hand-computed values.

    g(h) = μ / (R + h)²

    Key reference values:
        Surface (h=0):    μ/R²       = 4.9028695e12 / 1737400² = 1.6242 m/s²
        PDI (h=15240):    μ/(R+h)²   ≈ 1.6156 m/s²
        Apolune (h=110km):             ≈ 1.4389 m/s²
    """

    def test_surface_gravity(self) -> None:
        """Surface gravity from μ/R²."""
        g = gravity_at_altitude(0.0)
        expected = MU_MOON / (R_MOON * R_MOON)
        assert abs(g - expected) < 1e-6

    def test_pdi_altitude(self) -> None:
        """Gravity at PDI altitude (15.2 km)."""
        g = gravity_at_altitude(15_240.0)
        r = R_MOON + 15_240.0
        expected = MU_MOON / (r * r)
        assert abs(g - expected) < 1e-6
        # Should be less than surface gravity
        g_surface = gravity_at_altitude(0.0)
        assert g < g_surface

    def test_apolune_altitude(self) -> None:
        """Gravity at apolune (110 km) — significantly less than surface."""
        g = gravity_at_altitude(110_000.0)
        g_surface = gravity_at_altitude(0.0)
        # Should be about 11% less
        ratio = g / g_surface
        assert 0.88 < ratio < 0.90, f"g ratio at 110 km: {ratio:.4f}"

    def test_monotonically_decreasing(self) -> None:
        """Gravity must decrease with altitude."""
        altitudes = [0, 1000, 5000, 15240, 50000, 110000]
        gravities = [gravity_at_altitude(h) for h in altitudes]
        for i in range(len(gravities) - 1):
            assert gravities[i] > gravities[i + 1], \
                f"g({altitudes[i]}) = {gravities[i]:.6f} should be > " \
                f"g({altitudes[i+1]}) = {gravities[i+1]:.6f}"

    def test_negative_altitude_clamped(self) -> None:
        """Negative altitude should be clamped to 0 (surface)."""
        g_neg = gravity_at_altitude(-100.0)
        g_zero = gravity_at_altitude(0.0)
        assert abs(g_neg - g_zero) < 1e-10


class TestOrbitalVelocity:
    """Verify circular orbital velocity: v = sqrt(μ/r)."""

    def test_pdi_altitude(self) -> None:
        """Circular orbital velocity at PDI altitude."""
        v = orbital_velocity(15_240.0)
        r = R_MOON + 15_240.0
        expected = math.sqrt(MU_MOON / r)
        assert abs(v - expected) < 0.01

    def test_apolune(self) -> None:
        """Circular velocity at 110 km — should be less than at PDI."""
        v_pdi = orbital_velocity(15_240.0)
        v_apo = orbital_velocity(110_000.0)
        assert v_apo < v_pdi, "Higher orbit → lower velocity"

    def test_surface(self) -> None:
        """Surface circular velocity — about 1680 m/s."""
        v = orbital_velocity(0.0)
        expected = math.sqrt(MU_MOON / R_MOON)
        assert abs(v - expected) < 0.01
        assert 1670 < v < 1690, f"Surface v_circ: {v:.0f} m/s"


class TestVisViva:
    """Verify the vis-viva equation: v² = μ(2/r − 1/a).

    For the Apollo 11 descent orbit (110 km × 15.2 km):
        a = (r_apo + r_peri) / 2
        v_peri should be about 1695 m/s (close to 1690 reference)
    """

    def test_descent_orbit_perilune_velocity(self) -> None:
        """Vis-viva at perilune of descent orbit."""
        r_apo = R_MOON + DESCENT_ORBIT_APOLUNE
        r_peri = R_MOON + DESCENT_ORBIT_PERILUNE
        a = (r_apo + r_peri) / 2.0
        v = vis_viva_velocity(DESCENT_ORBIT_PERILUNE, a)
        # Should be close to 1690 m/s
        assert 1685 < v < 1700, f"v_peri: {v:.1f} m/s"

    def test_descent_orbit_apolune_velocity(self) -> None:
        """Velocity at apolune must be less than at perilune."""
        r_apo = R_MOON + DESCENT_ORBIT_APOLUNE
        r_peri = R_MOON + DESCENT_ORBIT_PERILUNE
        a = (r_apo + r_peri) / 2.0
        v_peri = vis_viva_velocity(DESCENT_ORBIT_PERILUNE, a)
        v_apo = vis_viva_velocity(DESCENT_ORBIT_APOLUNE, a)
        assert v_apo < v_peri, "Apolune velocity must be less than perilune"

    def test_circular_orbit(self) -> None:
        """For a circular orbit, vis-viva should match v_circ."""
        h = 50_000.0
        r = R_MOON + h
        a = r  # circular orbit: a = r
        v_vv = vis_viva_velocity(h, a)
        v_circ = orbital_velocity(h)
        assert abs(v_vv - v_circ) < 0.01, \
            f"Circular orbit: vis-viva {v_vv:.4f} vs v_circ {v_circ:.4f}"

    def test_energy_conservation(self) -> None:
        """Verify E = -μ/(2a) at both perilune and apolune."""
        r_apo = R_MOON + DESCENT_ORBIT_APOLUNE
        r_peri = R_MOON + DESCENT_ORBIT_PERILUNE
        a = (r_apo + r_peri) / 2.0

        v_peri = vis_viva_velocity(DESCENT_ORBIT_PERILUNE, a)
        v_apo = vis_viva_velocity(DESCENT_ORBIT_APOLUNE, a)

        E_peri = 0.5 * v_peri**2 - MU_MOON / r_peri
        E_apo = 0.5 * v_apo**2 - MU_MOON / r_apo
        E_theory = -MU_MOON / (2 * a)

        assert abs(E_peri - E_theory) < 1.0, f"E_peri: {E_peri:.0f} vs {E_theory:.0f}"
        assert abs(E_apo - E_theory) < 1.0, f"E_apo: {E_apo:.0f} vs {E_theory:.0f}"


class TestOrbitalElements:
    """Verify orbital element computation from state."""

    def test_descent_orbit(self) -> None:
        """Compute elements at PDI and verify they match the descent orbit."""
        r_apo = R_MOON + DESCENT_ORBIT_APOLUNE
        r_peri = R_MOON + DESCENT_ORBIT_PERILUNE
        a = (r_apo + r_peri) / 2.0
        v_peri = vis_viva_velocity(DESCENT_ORBIT_PERILUNE, a)

        orb = compute_orbital_elements(
            altitude=DESCENT_ORBIT_PERILUNE,
            vx=v_peri,
            vy=0.0,
        )

        # Semi-major axis should match
        assert abs(orb['semi_major_axis'] - a) / a < 0.001, \
            f"SMA: {orb['semi_major_axis']:.0f} vs {a:.0f}"

        # Periapsis altitude should be close to 15.2 km
        assert abs(orb['periapsis_alt'] - DESCENT_ORBIT_PERILUNE) < 100, \
            f"Periapsis: {orb['periapsis_alt']:.0f} vs {DESCENT_ORBIT_PERILUNE:.0f}"

        # Apoapsis altitude should be close to 110 km
        assert abs(orb['apoapsis_alt'] - DESCENT_ORBIT_APOLUNE) < 500, \
            f"Apoapsis: {orb['apoapsis_alt']:.0f} vs {DESCENT_ORBIT_APOLUNE:.0f}"

    def test_circular_orbit(self) -> None:
        """Circular orbit should have eccentricity ≈ 0."""
        h = 50_000.0
        v = orbital_velocity(h)
        orb = compute_orbital_elements(altitude=h, vx=v, vy=0.0)
        assert orb['eccentricity'] < 0.01, \
            f"Circular orbit e: {orb['eccentricity']:.6f}"

    def test_suborbital(self) -> None:
        """Low velocity should give negative periapsis (suborbital)."""
        orb = compute_orbital_elements(altitude=100.0, vx=10.0, vy=-5.0)
        assert orb['periapsis_alt'] < 0, "Suborbital trajectory should have negative periapsis"


class TestInverseSquareRK4:
    """Verify that RK4 with inverse-square gravity gives correct results."""

    VE = DPS_ISP * G0

    def test_freefall_inverse_square_vs_constant(self) -> None:
        """At low altitude, inverse-square and constant gravity should be very close."""
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        # Run with constant gravity
        y_const = 100.0
        vy_const = 0.0
        for _ in range(100):
            _, y_const, _, vy_const, _, _, _ = rk4_step_2d(
                0, y_const, 0, vy_const, 0, 0, LM_MASS_PDI,
                thrust_force=0.0, gravity=G_MOON,
                exhaust_velocity=self.VE, rcs_torque=0.0,
                dt=dt, dry_mass=dry, use_inverse_square=False,
            )

        # Run with inverse-square
        y_isq = 100.0
        vy_isq = 0.0
        for _ in range(100):
            _, y_isq, _, vy_isq, _, _, _ = rk4_step_2d(
                0, y_isq, 0, vy_isq, 0, 0, LM_MASS_PDI,
                thrust_force=0.0, gravity=G_MOON,
                exhaust_velocity=self.VE, rcs_torque=0.0,
                dt=dt, dry_mass=dry, use_inverse_square=True,
            )

        # At 100 m altitude, difference should be < 0.01%
        assert abs(y_const - y_isq) / abs(y_const) < 0.001, \
            f"Low-altitude: const={y_const:.4f} vs isq={y_isq:.4f}"

    def test_freefall_high_altitude_divergence(self) -> None:
        """At orbital altitude, inverse-square should give different results."""
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        # 15 km altitude, 100 steps = 1 second
        y_const = 15_240.0
        vy_const = 0.0
        for _ in range(100):
            _, y_const, _, vy_const, _, _, _ = rk4_step_2d(
                0, y_const, 0, vy_const, 0, 0, LM_MASS_PDI,
                thrust_force=0.0, gravity=G_MOON,
                exhaust_velocity=self.VE, rcs_torque=0.0,
                dt=dt, dry_mass=dry, use_inverse_square=False,
            )

        y_isq = 15_240.0
        vy_isq = 0.0
        for _ in range(100):
            _, y_isq, _, vy_isq, _, _, _ = rk4_step_2d(
                0, y_isq, 0, vy_isq, 0, 0, LM_MASS_PDI,
                thrust_force=0.0, gravity=G_MOON,
                exhaust_velocity=self.VE, rcs_torque=0.0,
                dt=dt, dry_mass=dry, use_inverse_square=True,
            )

        # Inverse-square gravity is slightly weaker at 15 km,
        # so the vehicle falls slightly less
        assert y_isq > y_const, \
            "Inverse-square: less gravity at altitude → less fall"

    def test_backward_compatible(self) -> None:
        """With use_inverse_square=False, results should be identical to Phase 3."""
        dt = 0.01
        dry = LM_MASS_PDI - PROPELLANT_MASS

        result_default = rk4_step_2d(
            0, 1000, 0, -50, 0.1, 0, LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST, gravity=G_MOON,
            exhaust_velocity=self.VE, rcs_torque=0.0,
            dt=dt, dry_mass=dry,
        )

        result_explicit = rk4_step_2d(
            0, 1000, 0, -50, 0.1, 0, LM_MASS_PDI,
            thrust_force=DPS_MAX_THRUST, gravity=G_MOON,
            exhaust_velocity=self.VE, rcs_torque=0.0,
            dt=dt, dry_mass=dry, use_inverse_square=False,
        )

        for i in range(7):
            assert abs(result_default[i] - result_explicit[i]) < 1e-10


class TestGuidancePrograms:
    """Verify guidance program logic."""

    def test_p63_retrograde_angle_at_pdi(self) -> None:
        """At PDI (vx=1690, vy≈0), retrograde should be ≈ -90° (thrust left)."""
        from guidance import compute_retrograde_angle
        angle = compute_retrograde_angle(vx=1690.0, vy=0.0)
        assert abs(angle - (-math.pi / 2)) < 0.01, \
            f"Retrograde at PDI: {math.degrees(angle):.1f}° vs -90°"

    def test_p63_retrograde_angle_vertical(self) -> None:
        """When only falling (vx=0, vy=-50), retrograde should be 0° (thrust up)."""
        from guidance import compute_retrograde_angle
        angle = compute_retrograde_angle(vx=0.0, vy=-50.0)
        assert abs(angle - 0.0) < 0.01, \
            f"Retrograde vertical: {math.degrees(angle):.1f}° vs 0°"

    def test_p63_full_thrust(self) -> None:
        """P63 should command full throttle."""
        from guidance import compute_p63_guidance
        cmd = compute_p63_guidance(vx=1690.0, vy=0.0, altitude=15240.0,
                                   mass=LM_MASS_PDI)
        assert cmd.target_throttle == 1.0
        assert cmd.auto_attitude is True
        assert cmd.auto_throttle is True

    def test_p66_auto_throttle_manual_attitude(self) -> None:
        """P66 should auto-throttle but leave attitude to pilot."""
        from guidance import compute_p66_guidance
        cmd = compute_p66_guidance(
            vx=0.0, vy=-1.0, altitude=100.0, mass=LM_MASS_PDI,
            rod_target=-0.9, previous_vy=-1.0, dt=0.01,
        )
        assert cmd.auto_throttle is True
        assert cmd.auto_attitude is False
        assert cmd.target_theta is None  # Pilot controls attitude

    def test_p67_all_manual(self) -> None:
        """P67 should not command anything."""
        from guidance import compute_p67_guidance
        cmd = compute_p67_guidance()
        assert cmd.auto_throttle is False
        assert cmd.auto_attitude is False
        assert cmd.target_theta is None
        assert cmd.target_throttle is None

    def test_auto_transition_p63_to_p64(self) -> None:
        """P63 hands off to P64 on VELOCITY, not on altitude alone.

        This test previously asserted that dropping below the altitude
        threshold at 500 m/s horizontal should enter P64. It shouldn't:
        being low while still travelling at 500 m/s is not "approach",
        it is a failed braking phase, and the end-to-end mission test
        showed that handing off there puts the vehicle into the surface
        at several hundred m/s. The altitude condition now only applies
        once the vehicle is also reasonably slow.
        """
        from guidance import GuidanceState, GuidanceMode, update_guidance
        from constants import P63_TO_P64_ALTITUDE, P63_TO_P64_VELOCITY
        gs = GuidanceState(mode=GuidanceMode.P63, auto_transition=True)

        # High and fast — stay in P63
        update_guidance(0, P63_TO_P64_ALTITUDE + 100, 500, -10, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P63

        # Low but STILL FAST — must NOT hand off; braking is not done
        update_guidance(0, P63_TO_P64_ALTITUDE - 100, 500, -10, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P63

        # Slow enough — now it is a real approach
        update_guidance(0, P63_TO_P64_ALTITUDE - 100,
                        P63_TO_P64_VELOCITY - 10, -10, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P64

    def test_p63_to_p64_handoff_is_survivable(self) -> None:
        """Whatever the thresholds are, the handoff state must be flyable."""
        from guidance import GuidanceState, GuidanceMode, update_guidance
        gs = GuidanceState(mode=GuidanceMode.P63, auto_transition=True)
        # A fast, low state must never be classified as "approach".
        update_guidance(0, 500.0, 800.0, -80.0, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P63

    def test_auto_transition_p64_to_p66(self) -> None:
        """P64 should transition to P66 when altitude drops below 150 m."""
        from guidance import GuidanceState, GuidanceMode, update_guidance
        from constants import P64_TO_P66_ALTITUDE
        gs = GuidanceState(mode=GuidanceMode.P64, auto_transition=True)

        # Above threshold
        update_guidance(0, P64_TO_P66_ALTITUDE + 50, 10, -5, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P64

        # Below threshold
        update_guidance(0, P64_TO_P66_ALTITUDE - 10, 10, -5, LM_MASS_PDI, gs, 0.01)
        assert gs.mode == GuidanceMode.P66

    def test_p66_rod_throttle_increases_for_fast_descent(self) -> None:
        """If descending faster than target, P66 should increase throttle."""
        from guidance import compute_p66_guidance
        # Target: -0.9 m/s, actual: -3.0 m/s (too fast)
        cmd = compute_p66_guidance(
            vx=0.0, vy=-3.0, altitude=100.0, mass=LM_MASS_PDI,
            rod_target=-0.9, previous_vy=-3.0, dt=0.01,
        )
        # Hover throttle is about mass*g / max_thrust
        g = gravity_at_altitude(100.0)
        hover_throttle = LM_MASS_PDI * g / DPS_MAX_THRUST
        # Should be above hover to slow descent
        assert cmd.target_throttle > hover_throttle, \
            f"Throttle {cmd.target_throttle:.3f} should be > hover {hover_throttle:.3f}"


# ============================================================================
# PHASE 5 TESTS — AGC & DSKY INTERFACE
# ============================================================================

class TestAGCRegisters:
    """Verify DSKY register updates for each noun."""

    def _make_agc_and_update(self, noun: int):
        """Create an AGC, set noun, and update with test data."""
        from agc import AGC
        agc = AGC()
        agc.set_verb_noun(16, noun)
        agc.update_display(
            altitude=5000.0,
            velocity=-20.0,
            vx=100.0,
            vy=-20.0,
            theta=0.1,
            omega=0.05,
            mass=12000.0,
            delta_v=1500.0,
            burn_time=300.0,
            propellant=5000.0,
            propellant_max=8200.0,
            x=1000.0,
            landing_site_x=0.0,
            time_elapsed=400.0,
        )
        return agc

    def test_noun_62(self) -> None:
        """N62: velocity, altitude rate, altitude."""
        agc = self._make_agc_and_update(62)
        r = agc.registers
        assert r.noun == 62
        assert abs(r.r1_value - math.sqrt(100**2 + 20**2)) < 0.5
        assert abs(r.r2_value - (-20.0)) < 0.01
        assert abs(r.r3_value - 5000.0) < 0.01

    def test_noun_63(self) -> None:
        """N63: altitude, altitude rate, downrange."""
        agc = self._make_agc_and_update(63)
        r = agc.registers
        assert abs(r.r1_value - 5000.0) < 0.01
        assert abs(r.r2_value - (-20.0)) < 0.01
        assert abs(r.r3_value - 1000.0) < 0.01

    def test_noun_64(self) -> None:
        """N64: Δv, burn time, propellant %."""
        agc = self._make_agc_and_update(64)
        r = agc.registers
        assert abs(r.r1_value - 1500.0) < 0.01
        assert abs(r.r2_value - 300.0) < 0.01
        expected_pct = 5000.0 / 8200.0 * 100.0
        assert abs(r.r3_value - expected_pct) < 0.1

    def test_noun_68(self) -> None:
        """N68: range, velocity, altitude rate."""
        agc = self._make_agc_and_update(68)
        r = agc.registers
        assert abs(r.r1_value - 1000.0) < 0.01
        assert abs(r.r2_value - math.sqrt(100**2 + 20**2)) < 0.5
        assert abs(r.r3_value - (-20.0)) < 0.01

    def test_noun_69(self) -> None:
        """N69: theta(deg), omega(deg/s), MOI."""
        agc = self._make_agc_and_update(69)
        r = agc.registers
        assert abs(r.r1_value - math.degrees(0.1)) < 0.1
        assert abs(r.r2_value - math.degrees(0.05)) < 0.1


class TestAGCNounCycling:
    """Verify noun cycling through the sequence."""

    def test_cycle_sequence(self) -> None:
        """Cycling should go 62→63→64→68→69→62."""
        from agc import AGC
        agc = AGC()
        agc.set_verb_noun(16, 62)
        expected = [63, 64, 68, 69, 62]
        for exp in expected:
            agc.cycle_noun()
            assert agc.registers.noun == exp

    def test_cycle_from_unknown(self) -> None:
        """Cycling from unknown noun should reset to 62."""
        from agc import AGC
        agc = AGC()
        agc.registers.noun = 99
        agc.cycle_noun()
        assert agc.registers.noun == 62


class TestAGCAlarms:
    """Verify program alarm system."""

    def test_trigger_alarm(self) -> None:
        """Triggering an alarm should set PROG light."""
        from agc import AGC
        agc = AGC()
        assert agc.registers.prog_light is False
        agc.trigger_alarm(1202, "EXEC OVERFLOW", 100.0)
        assert agc.registers.prog_light is True
        assert len(agc.alarms) == 1
        assert agc.alarms[0].code == 1202

    def test_acknowledge_alarm(self) -> None:
        """Acknowledging should clear the alarm and PROG light."""
        from agc import AGC
        agc = AGC()
        agc.trigger_alarm(1202, "EXEC OVERFLOW", 100.0)
        agc.acknowledge_alarm()
        assert agc.alarms[0].acknowledged is True
        assert agc.registers.prog_light is False

    def test_multiple_alarms(self) -> None:
        """Multiple alarms: only clear PROG when all acknowledged."""
        from agc import AGC
        agc = AGC()
        agc.trigger_alarm(1202, "EXEC OVERFLOW", 100.0)
        agc._last_alarm_time = 0
        agc.trigger_alarm(1201, "NO VAC AREAS", 150.0)
        agc.acknowledge_alarm()
        assert agc.registers.prog_light is True
        agc.acknowledge_alarm()
        assert agc.registers.prog_light is False

    def test_get_active_alarm(self) -> None:
        """get_active_alarm returns most recent unacknowledged."""
        from agc import AGC
        agc = AGC()
        assert agc.get_active_alarm() is None
        agc.trigger_alarm(1202, "EXEC OVERFLOW", 100.0)
        assert agc.get_active_alarm().code == 1202
        agc.acknowledge_alarm()
        assert agc.get_active_alarm() is None


class TestAGCProgramTracking:
    """Verify program number sync."""

    def test_set_program(self) -> None:
        from agc import AGC
        agc = AGC()
        agc.set_program(63)
        assert agc.registers.prog == 63
        agc.set_program(66)
        assert agc.registers.prog == 66

    def test_tracker_light(self) -> None:
        """TRACKER light should activate below 12 km."""
        from agc import AGC
        agc = AGC()
        agc.update_display(
            altitude=15000, velocity=0, vx=0, vy=0, theta=0, omega=0,
            mass=15000, delta_v=2000, burn_time=500,
            propellant=8000, propellant_max=8200,
            x=0, landing_site_x=0, time_elapsed=100,
        )
        assert agc.registers.tracker is False
        agc.update_display(
            altitude=11000, velocity=0, vx=0, vy=0, theta=0, omega=0,
            mass=15000, delta_v=2000, burn_time=500,
            propellant=8000, propellant_max=8200,
            x=0, landing_site_x=0, time_elapsed=200,
        )
        assert agc.registers.tracker is True

    def test_format_register(self) -> None:
        """Verify DSKY number formatting."""
        from agc import AGC
        agc = AGC()
        assert "+" in agc.format_register(123.4)
        assert "-" in agc.format_register(-50.3)
        assert "15240" in agc.format_register(15240.0)


# ============================================================================
# PHASE 6 TESTS — LANDING RADAR & TERRAIN
# ============================================================================

class TestLunarTerrain:
    """Verify procedural terrain generation."""

    def test_landing_pad_flat(self) -> None:
        """Landing pad area should be perfectly flat."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0, pad_half_width=50.0)
        pad_h = t.height_at(0.0)
        for x in [-40, -20, 0, 20, 40]:
            assert t.height_at(x) == pad_h, \
                f"Pad not flat at x={x}: {t.height_at(x)} != {pad_h}"

    def test_terrain_varies(self) -> None:
        """Terrain outside the pad should not be perfectly flat."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0)
        heights = [t.height_at(x) for x in range(200, 5000, 100)]
        # Not all the same
        assert len(set(heights)) > 1, "Terrain has no variation"

    def test_terrain_deterministic(self) -> None:
        """Same seed should produce same terrain."""
        from terrain import LunarTerrain
        t1 = LunarTerrain(seed=42)
        t2 = LunarTerrain(seed=42)
        for x in [0, 100, 500, 1000, 5000]:
            assert t1.height_at(x) == t2.height_at(x)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different terrain."""
        from terrain import LunarTerrain
        t1 = LunarTerrain(seed=1)
        t2 = LunarTerrain(seed=2)
        # At least some points differ
        diffs = sum(
            1 for x in range(0, 5000, 100)
            if abs(t1.height_at(x) - t2.height_at(x)) > 0.01
        )
        assert diffs > 0

    def test_slope_at_pad(self) -> None:
        """Slope on the flat landing pad should be ~0."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0, pad_half_width=50.0)
        slope = t.slope_at(0.0)
        assert abs(slope) < 0.01, f"Pad slope too large: {math.degrees(slope):.2f}°"

    def test_pad_is_safe_landing(self) -> None:
        """Landing pad should be classified as safe."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0, pad_half_width=50.0)
        assert t.is_safe_landing(0.0)

    def test_terrain_profile(self) -> None:
        """get_terrain_profile should return correct number of points."""
        from terrain import LunarTerrain
        t = LunarTerrain()
        profile = t.get_terrain_profile(0, 1000, num_points=100)
        assert len(profile) == 101  # num_points + 1
        for x, h in profile:
            assert isinstance(x, float)
            assert isinstance(h, float)

    def test_smooth_pad_transition(self) -> None:
        """Heights should transition smoothly at pad edges."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0, pad_half_width=50.0)
        # Just inside pad
        h_inside = t.height_at(49.0)
        # Just outside pad (in blend zone)
        h_blend = t.height_at(55.0)
        # The transition should be smooth, not a cliff
        # The blend zone is 20m, so at +55 we're 5m into blend
        assert abs(h_blend - h_inside) < 20.0, \
            "Pad-to-terrain transition too abrupt"


class TestLandingRadar:
    """Verify radar lock, measurements, and noise."""

    def test_out_of_range(self) -> None:
        """Radar should not lock above max range."""
        from terrain import LunarTerrain
        from radar import LandingRadar, RADAR_MAX_RANGE
        t = LunarTerrain()
        r = LandingRadar(terrain=t)
        reading = r.update(0.0, RADAR_MAX_RANGE + 1000, 0.0, 0.0, 1.0)
        assert not reading.range_valid
        assert not r.is_locked

    def test_lock_acquisition(self) -> None:
        """Radar should lock after being in range for RADAR_LOCK_TIME."""
        from terrain import LunarTerrain
        from radar import LandingRadar, RADAR_LOCK_TIME
        t = LunarTerrain()
        r = LandingRadar(terrain=t)
        # First update: start lock timer
        r.update(0.0, 5000.0, 0.0, 0.0, 0.0)
        assert not r.is_locked
        # After lock time: should lock
        r.update(0.0, 5000.0, 0.0, 0.0, RADAR_LOCK_TIME + 0.1)
        assert r.is_locked

    def test_locked_readings_valid(self) -> None:
        """Once locked, readings should be valid."""
        from terrain import LunarTerrain
        from radar import LandingRadar, RADAR_LOCK_TIME
        t = LunarTerrain()
        r = LandingRadar(terrain=t)
        r.update(0.0, 5000.0, 0.0, 0.0, 0.0)
        reading = r.update(0.0, 5000.0, 100.0, -10.0, RADAR_LOCK_TIME + 0.1)
        assert reading.range_valid
        assert reading.velocity_valid
        assert reading.data_good

    def test_range_accuracy(self) -> None:
        """Radar range should be within spec of true altitude."""
        from terrain import LunarTerrain
        from radar import LandingRadar, RADAR_LOCK_TIME
        t = LunarTerrain()
        r = LandingRadar(terrain=t)
        r.update(0.0, 5000.0, 0.0, 0.0, 0.0)
        readings = []
        for i in range(50):
            rd = r.update(0.0, 5000.0, 0.0, 0.0, RADAR_LOCK_TIME + 1.0 + i)
            if rd.range_valid:
                readings.append(rd.range_m)
        assert len(readings) > 0
        true_agl = r.altitude_agl(0.0, 5000.0)
        for meas in readings:
            error_pct = abs(meas - true_agl) / true_agl
            assert error_pct < 0.05, \
                f"Range error {error_pct*100:.1f}% exceeds 5%"

    def test_altitude_agl(self) -> None:
        """altitude_agl should return true height above terrain."""
        from terrain import LunarTerrain
        from radar import LandingRadar
        t = LunarTerrain(landing_site_x=0.0)
        r = LandingRadar(terrain=t)
        pad_h = t.height_at(0.0)
        agl = r.altitude_agl(0.0, 5000.0)
        assert abs(agl - (5000.0 - pad_h)) < 0.01

    def test_unlock_above_range(self) -> None:
        """Radar should unlock if vehicle goes above max range."""
        from terrain import LunarTerrain
        from radar import LandingRadar, RADAR_LOCK_TIME, RADAR_MAX_RANGE
        t = LunarTerrain()
        r = LandingRadar(terrain=t)
        # Lock
        r.update(0.0, 5000.0, 0.0, 0.0, 0.0)
        r.update(0.0, 5000.0, 0.0, 0.0, RADAR_LOCK_TIME + 0.1)
        assert r.is_locked
        # Go above range
        r.update(0.0, RADAR_MAX_RANGE + 500, 0.0, 0.0, RADAR_LOCK_TIME + 1.0)
        assert not r.is_locked


class TestTerrainAwareLanding:
    """Verify terrain-relative landing detection."""

    def test_landing_on_flat_pad(self) -> None:
        """Landing on the pad should succeed with good velocities."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0)
        pad_h = t.height_at(0.0)
        status, violations = check_landing_2d(
            y=pad_h, vx=0.3, vy=-0.5,
            theta=0.01, propellant=100.0,
            terrain_height=pad_h,
        )
        assert status == "landed"

    def test_crash_below_terrain(self) -> None:
        """Vehicle below terrain height at high speed = crash."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0)
        terrain_h = t.height_at(500.0)
        status, violations = check_landing_2d(
            y=terrain_h, vx=0.0, vy=-10.0,
            theta=0.0, propellant=100.0,
            terrain_height=terrain_h,
        )
        assert status == "crashed"
        assert any("Vertical" in v for v in violations)

    def test_still_flying_above_terrain(self) -> None:
        """Vehicle above terrain height should still be flying."""
        from terrain import LunarTerrain
        t = LunarTerrain(landing_site_x=0.0)
        terrain_h = t.height_at(0.0)
        status, violations = check_landing_2d(
            y=terrain_h + 100.0, vx=100.0, vy=-20.0,
            theta=0.5, propellant=1000.0,
            terrain_height=terrain_h,
        )
        assert status == "flying"

    def test_backward_compat_no_terrain(self) -> None:
        """Without terrain_height, landing check works as before (y=0)."""
        status, violations = check_landing_2d(
            y=0.0, vx=0.3, vy=-0.5,
            theta=0.01, propellant=100.0,
        )
        assert status == "landed"
