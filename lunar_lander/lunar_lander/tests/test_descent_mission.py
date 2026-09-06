"""End-to-end powered descent under the autopilot.

WHY THIS FILE EXISTS
════════════════════
Every other test in this suite checks a *function*: does the integrator
conserve energy, does the retrograde angle come out right, does the
throttle clamp. All of them passed while the mission itself was
impossible — the vehicle flew P63, P64 and P66 exactly as written and
hit the Moon at 1,235 m/s.

Unit tests prove the parts are self-consistent. Only flying the whole
profile proves the parts add up to a landing. Enabling round-Moon
physics is what finally exposed this: the flat model had been hiding a
guidance stack that never worked.

Bugs this file caught, none of which any unit test could see:

  1. P63 commanded pure retrograde thrust — no vertical component, so
     the vehicle free-fell at 150+ m/s while it braked.
  2. P63 -> P64 transitioned on altitude OR velocity, so it entered
     "approach" at 1,239 m/s.
  3. P64 tilted the thrust WITH the horizontal velocity instead of
     against it — horizontal speed climbed 150 -> 190 m/s.
  4. P64 commanded hover-throttle + 0.05, which cannot arrest a 59 m/s
     descent.
  5. P66's PD loop differentiated its own output at the physics
     timestep and chattered, holding -0.6 m/s when -3.0 was commanded
     and running the tanks dry at 60 m.
  6. P66 released attitude with a residual rate; the attitude-hold law
     damps rate but never nulls angle, so a fraction of a degree
     integrated into 11 m/s of lateral drift.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import physics as ph                                     # noqa: E402
from constants import (                                  # noqa: E402
    ATTITUDE_HOLD_RATE,
    G_MOON,
    MAX_HORIZONTAL_SPEED,
    MAX_VERTICAL_SPEED,
    PHASE4_START_ANGLE,
    PHASE4_START_OMEGA,
    PHASE4_START_VX,
    PHASE4_START_VY,
    PHASE4_START_X,
    PHASE4_START_Y,
    PHYSICS_DT,
)
from guidance import (                                   # noqa: E402
    GuidanceMode,
    GuidanceState,
    update_guidance,
)
from vehicle import LunarModule                          # noqa: E402

ATTITUDE_SLEW_RATE = 0.1      # rad/s — matches the game's autopilot slew
MISSION_TIMEOUT = 1800.0      # s


class DescentResult:
    """Outcome of one full powered descent."""

    def __init__(self, t, x, y, vx, vy, theta, propellant, mode, modes_seen):
        self.t = t
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.propellant = propellant
        self.mode = mode
        self.modes_seen = modes_seen

    @property
    def landed(self) -> bool:
        return (abs(self.vy) <= MAX_VERTICAL_SPEED
                and abs(self.vx) <= MAX_HORIZONTAL_SPEED)


def fly_descent(curvilinear: bool = True) -> DescentResult:
    """Fly the whole profile from PDI to the surface with no pilot input.

    The attitude model mirrors main.py: while guidance owns attitude it
    slews toward the commanded angle at a bounded rate; when guidance
    releases attitude the hold law damps the residual rate.
    """
    lm = LunarModule()
    gs = GuidanceState(mode=GuidanceMode.P63, auto_transition=True)

    x, y = PHASE4_START_X, PHASE4_START_Y
    vx, vy = PHASE4_START_VX, PHASE4_START_VY
    theta, omega = PHASE4_START_ANGLE, PHASE4_START_OMEGA
    mass = lm.mass
    t, dt = 0.0, PHYSICS_DT
    modes_seen = [gs.mode]

    while t < MISSION_TIMEOUT:
        cmd = update_guidance(x, y, vx, vy, mass, gs, dt)
        if gs.mode not in modes_seen:
            modes_seen.append(gs.mode)

        if cmd.auto_throttle and cmd.target_throttle is not None:
            lm.throttle = cmd.target_throttle

        if cmd.auto_attitude and cmd.target_theta is not None:
            err = cmd.target_theta - theta
            max_step = ATTITUDE_SLEW_RATE * dt
            if abs(err) > max_step:
                err = math.copysign(max_step, err)
            theta += err
            omega = err / dt
        else:
            omega *= math.exp(-ATTITUDE_HOLD_RATE * dt)

        mode_kwargs = ({"curvilinear": True} if curvilinear
                       else {"use_inverse_square": True})
        x, y, vx, vy, theta, omega, mass = ph.rk4_step_2d(
            x, y, vx, vy, theta, omega, mass,
            lm.thrust_force, G_MOON, lm.exhaust_velocity,
            lm.rcs_torque_value, dt,
            dry_mass=lm.dry_mass, **mode_kwargs
        )
        lm.sync_mass_from_physics(mass)
        t += dt
        if y <= 0.0:
            break

    return DescentResult(t, x, y, vx, vy, theta,
                         mass - lm.dry_mass, gs.mode, modes_seen)


@pytest.fixture(scope="module")
def descent():
    return fly_descent(curvilinear=True)


class TestPoweredDescentMission:
    """The whole profile, PDI to touchdown, hands off."""

    def test_soft_landing(self, descent):
        assert descent.landed, (
            f"crashed: vx={descent.vx:.2f} vy={descent.vy:.2f} "
            f"(limits {MAX_HORIZONTAL_SPEED}/{MAX_VERTICAL_SPEED})"
        )

    def test_touchdown_velocities_within_gear_limits(self, descent):
        assert abs(descent.vy) <= MAX_VERTICAL_SPEED
        assert abs(descent.vx) <= MAX_HORIZONTAL_SPEED

    def test_reaches_the_surface(self, descent):
        assert descent.y <= 0.0
        assert descent.t < MISSION_TIMEOUT

    def test_propellant_remains_at_touchdown(self, descent):
        """Landing on fumes is still landing, but empty is a crash."""
        assert descent.propellant > 0.0

    def test_flies_the_full_program_sequence(self, descent):
        assert descent.modes_seen[:3] == [
            GuidanceMode.P63, GuidanceMode.P64, GuidanceMode.P66
        ]

    def test_upright_at_touchdown(self, descent):
        """A tipped-over lander does not survive contact."""
        assert abs(math.degrees(descent.theta)) < 10.0

    def test_descent_duration_is_physically_plausible(self, descent):
        """Apollo 11 PDI to touchdown was roughly 12 minutes."""
        assert 400.0 < descent.t < 1000.0


class TestDescentUnderBothGravityModels:
    """Curvilinear and Cartesian must both fly it — same physics."""

    def test_cartesian_central_gravity_also_lands(self):
        result = fly_descent(curvilinear=False)
        assert result.landed, (
            f"crashed: vx={result.vx:.2f} vy={result.vy:.2f}"
        )

    def test_both_models_agree_on_outcome(self):
        curv = fly_descent(curvilinear=True)
        cart = fly_descent(curvilinear=False)
        assert curv.landed == cart.landed
        assert curv.vy == pytest.approx(cart.vy, abs=0.5)
