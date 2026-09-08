"""
Physics engine for the Lunar Module descent simulator.

This module contains the core numerical integration and dynamics.
All functions are PURE — state in, result out, no side effects.
This makes them deterministic and unit-testable.

Physics background
──────────────────
Phase 1: Point mass in 1D, constant gravity, constant vehicle mass.
Phase 2: Variable-mass dynamics — propellant burns off, changing the
         vehicle's response to thrust. The state vector grows from
         [h, v] to [h, v, m].
Phase 3: Two-dimensional dynamics with rotation and RCS.
Phase 4: Inverse-square gravity, orbital mechanics. Gravity is no
         longer constant — it decreases with altitude as μ/(R+h)².
         This matters at orbital altitudes (15 km) and is computed
         at each RK4 sub-step for correctness.

The variable-mass rocket equation:
    The vehicle's mass m decreases as propellant is consumed:
        dm/dt = −F_thrust / v_e

    where v_e = Isp × g0 is the effective exhaust velocity.

    Because mass appears in the denominator of F=ma, thrust acceleration
    a = F/m INCREASES as mass decreases. The same engine becomes more
    effective as the vehicle empties — this is the fundamental dynamic
    that makes rockets work (and makes them hard to fly).

    The total velocity change possible from a given amount of propellant
    is the Tsiolkovsky rocket equation:
        Δv = v_e × ln(m_initial / m_final)

    This is logarithmic, not linear: the last 10% of propellant gives
    much less Δv than the first 10%. This is the "tyranny of the rocket
    equation" — payload fraction is exponentially punished.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

from constants import G0, DPS_ISP, MU_MOON, R_MOON


@dataclass
class SimState:
    """Complete simulation state at one instant in time.

    Attributes:
        altitude:  Height above lunar surface [m]. Positive up.
        velocity:  Vertical speed [m/s]. Positive up.
        time:      Mission elapsed time [s].
        thrust_on: Whether the DPS engine is currently firing.
        mass:      Total vehicle mass [kg] (Phase 2+). Decreases as
                   propellant burns. Affects thrust acceleration: a = F/m.
        throttle:  DPS throttle setting (Phase 2+). Range: 0.0 (off),
                   0.10–0.60 (continuous), or 1.00 (fixed full thrust).
    """
    altitude: float
    velocity: float
    time: float = 0.0
    thrust_on: bool = False
    mass: float = 0.0       # Phase 2: total vehicle mass [kg]
    throttle: float = 0.0   # Phase 2: throttle setting [0.0 - 1.0]


# ============================================================================
# PHASE 1 FUNCTIONS (preserved for backward compatibility and tests)
# ============================================================================

def derivatives(
    altitude: float,
    velocity: float,
    thrust_accel: float,
    gravity: float,
) -> np.ndarray:
    """Compute time-derivatives for constant-mass dynamics (Phase 1).

    State: y = [altitude, velocity]
    Derivatives: [dh/dt, dv/dt] = [v, a_thrust − g]

    See Phase 1 docstrings for full derivation.
    """
    dh_dt = velocity
    dv_dt = thrust_accel - gravity
    return np.array([dh_dt, dv_dt], dtype=np.float64)


def rk4_step(
    altitude: float,
    velocity: float,
    thrust_accel: float,
    gravity: float,
    dt: float,
) -> tuple[float, float]:
    """Advance [altitude, velocity] by one RK4 step (Phase 1, constant mass).

    See Phase 1 docstrings for the full RK4 derivation.
    """
    y = np.array([altitude, velocity], dtype=np.float64)

    k1 = derivatives(y[0], y[1], thrust_accel, gravity)
    y2 = y + 0.5 * dt * k1
    k2 = derivatives(y2[0], y2[1], thrust_accel, gravity)
    y3 = y + 0.5 * dt * k2
    k3 = derivatives(y3[0], y3[1], thrust_accel, gravity)
    y4 = y + dt * k3
    k4 = derivatives(y4[0], y4[1], thrust_accel, gravity)

    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return float(y_new[0]), float(y_new[1])


# ============================================================================
# PHASE 2 FUNCTIONS — VARIABLE-MASS DYNAMICS
# ============================================================================

def derivatives_with_mass(
    altitude: float,
    velocity: float,
    mass: float,
    thrust_force: float,
    gravity: float,
    exhaust_velocity: float,
) -> np.ndarray:
    """Compute time-derivatives for variable-mass dynamics.

    THE EXTENDED STATE VECTOR
    ─────────────────────────
    Phase 2 adds mass as a dynamic variable. The state is now:

        y = [h, v, m]

    with derivatives:

        dh/dt = v                                    ... velocity = rate of height change
        dv/dt = F_thrust / m  −  g                   ... Newton's second law for variable mass
        dm/dt = −F_thrust / v_e                      ... propellant consumption

    WHY MASS IS INSIDE THE DERIVATIVE FUNCTION
    ───────────────────────────────────────────
    You might think: "just subtract the fuel after the step." That would be
    wrong. Within a single RK4 step, the k₂/k₃/k₄ evaluations occur at
    different masses, which changes the acceleration F/m. If we froze mass
    during the step, we'd overestimate thrust at the start and underestimate
    it at the end (or vice versa), introducing a systematic error.

    With dt = 0.01s and ṁ ≈ 14.8 kg/s, mass changes by ~0.15 kg per step
    out of ~15,000 kg — a 0.001% change. So the error would be tiny in
    Phase 2. But when we add inverse-square gravity in Phase 4, both mass
    AND gravity change within a step, and the errors compound. Getting it
    right now prevents debugging later.

    MASS FLOW EQUATION
    ──────────────────
    dm/dt = −F / v_e = −F / (Isp × g0)

    The minus sign means mass DECREASES when thrust is positive.
    v_e is the effective exhaust velocity: the speed at which propellant
    leaves the nozzle. For the DPS: v_e = 311 × 9.80665 ≈ 3,050 m/s.

    The fuel consumption rate at full thrust:
        ṁ = 45,040 / 3,050 ≈ 14.77 kg/s

    That's about 14.8 kg every second at full power.

    Args:
        altitude:         Current altitude [m]. Not used in Phase 2 (constant g)
                          but included for Phase 4 compatibility.
        velocity:         Current vertical velocity [m/s].
        mass:             Current total vehicle mass [kg].
        thrust_force:     Engine thrust [N]. Zero if engine is off.
                          In Phase 2: throttle × DPS_MAX_THRUST.
        gravity:          Local gravitational acceleration [m/s²]. Positive value.
        exhaust_velocity: Effective exhaust velocity v_e = Isp × g0 [m/s].

    Returns:
        np.ndarray of shape (3,): [dh/dt, dv/dt, dm/dt].
    """
    dh_dt = velocity

    # Thrust acceleration: F/m
    # Guard against zero/negative mass (shouldn't happen in normal operation,
    # but protect the integrator during intermediate RK4 evaluations)
    if mass > 0.0 and thrust_force > 0.0:
        dv_dt = thrust_force / mass - gravity
        # Mass flow rate: propellant consumed per second
        # Negative because mass decreases when thrust is positive
        dm_dt = -thrust_force / exhaust_velocity
    elif mass > 0.0:
        # No thrust, only gravity
        dv_dt = -gravity
        dm_dt = 0.0
    else:
        # No mass (shouldn't happen) — just gravity, no thrust
        dv_dt = -gravity
        dm_dt = 0.0

    return np.array([dh_dt, dv_dt, dm_dt], dtype=np.float64)


def rk4_step_with_mass(
    altitude: float,
    velocity: float,
    mass: float,
    thrust_force: float,
    gravity: float,
    exhaust_velocity: float,
    dt: float,
    dry_mass: float = 0.0,
) -> tuple[float, float, float]:
    """Advance [altitude, velocity, mass] by one RK4 step.

    VARIABLE-MASS RK4
    ──────────────────
    This is the same RK4 algorithm as Phase 1, but now applied to a
    3-component state vector [h, v, m]. The key difference: each of the
    four derivative evaluations uses a DIFFERENT mass, because mass
    changes during the step. This means:

    - k₁ uses mass m (start of step)
    - k₂ uses mass m + dt/2 × dm₁ (predicted midpoint mass)
    - k₃ uses mass m + dt/2 × dm₂ (improved midpoint mass)
    - k₄ uses mass m + dt × dm₃ (predicted end-of-step mass)

    The thrust acceleration F/m is recomputed at each evaluation point
    using the local mass. This correctly captures the fact that the
    vehicle accelerates faster as it gets lighter.

    FUEL EXHAUSTION HANDLING
    ────────────────────────
    If mass drops to dry_mass during a sub-step, thrust is zeroed for
    subsequent evaluations. This prevents the integrator from producing
    negative propellant. The main loop performs the definitive exhaustion
    check after each step.

    Args:
        altitude:         Current altitude [m].
        velocity:         Current vertical velocity [m/s].
        mass:             Current total vehicle mass [kg].
        thrust_force:     Engine thrust [N].
        gravity:          Gravitational acceleration [m/s²].
        exhaust_velocity: v_e = Isp × g0 [m/s].
        dt:               Timestep [s].
        dry_mass:         Vehicle mass with no propellant [kg]. Used to
                          prevent mass from going below the structural mass.

    Returns:
        Tuple (new_altitude, new_velocity, new_mass).
    """
    y = np.array([altitude, velocity, mass], dtype=np.float64)

    def _safe_thrust(m: float) -> float:
        """Return thrust_force if propellant remains, else 0."""
        if m <= dry_mass or thrust_force <= 0.0:
            return 0.0
        return thrust_force

    # k₁: derivative at the start
    k1 = derivatives_with_mass(
        y[0], y[1], y[2], _safe_thrust(y[2]), gravity, exhaust_velocity,
    )

    # k₂: derivative at the midpoint using k₁
    y2 = y + 0.5 * dt * k1
    y2[2] = max(y2[2], dry_mass)  # clamp mass floor
    k2 = derivatives_with_mass(
        y2[0], y2[1], y2[2], _safe_thrust(y2[2]), gravity, exhaust_velocity,
    )

    # k₃: derivative at the midpoint using k₂
    y3 = y + 0.5 * dt * k2
    y3[2] = max(y3[2], dry_mass)
    k3 = derivatives_with_mass(
        y3[0], y3[1], y3[2], _safe_thrust(y3[2]), gravity, exhaust_velocity,
    )

    # k₄: derivative at the end using k₃
    y4 = y + dt * k3
    y4[2] = max(y4[2], dry_mass)
    k4 = derivatives_with_mass(
        y4[0], y4[1], y4[2], _safe_thrust(y4[2]), gravity, exhaust_velocity,
    )

    # Weighted average (Simpson's rule)
    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Final mass clamp
    y_new[2] = max(y_new[2], dry_mass)

    return float(y_new[0]), float(y_new[1]), float(y_new[2])


def compute_delta_v_remaining(
    mass_current: float,
    mass_dry: float,
    isp: float = DPS_ISP,
    g0: float = G0,
) -> float:
    """Compute remaining Δv using the Tsiolkovsky rocket equation.

    THE ROCKET EQUATION — THE MOST IMPORTANT EQUATION IN SPACEFLIGHT
    ─────────────────────────────────────────────────────────────────
    Δv = v_e × ln(m₀ / m_f) = Isp × g0 × ln(m₀ / m_f)

    where:
        v_e  = effective exhaust velocity = Isp × g0
        m₀   = current total mass (structure + remaining propellant)
        m_f   = dry mass (structure only, propellant = 0)
        ln   = natural logarithm

    This equation is LOGARITHMIC, which has profound consequences:

    1. Diminishing returns: each additional m/s of Δv costs exponentially
       more propellant. Going from 0 to 1000 m/s takes X kg of fuel.
       Going from 1000 to 2000 m/s takes X × e^(1000/v_e) times more.

    2. The "tyranny" of the rocket equation: to carry more propellant,
       you need a bigger tank, which needs more propellant to lift...
       This recursive dependency makes single-stage-to-orbit barely possible
       and interplanetary missions require staging or refueling.

    3. Why the LM's mass fraction matters: the DPS has 8,200 kg propellant
       in a 15,100 kg vehicle. That's a propellant fraction of 54%.
       Δv = 3050 × ln(15100/6900) ≈ 2,390 m/s.
       If we could somehow carry 12,000 kg of propellant (80% fraction):
       Δv = 3050 × ln(15100/3100) ≈ 4,860 m/s — twice as much!
       Every percentage point of mass fraction is worth real capability.

    AT PDI:
        Δv = 311 × 9.80665 × ln(15100 / 6900)
           = 3049.87 × ln(2.1884)
           = 3049.87 × 0.7831
           ≈ 2,388 m/s

    This is the budget for the entire descent. The actual Apollo 11
    descent used about 2,050 m/s of this, leaving margin.

    Args:
        mass_current: Current total vehicle mass [kg].
        mass_dry:     Mass with all propellant expended [kg].
        isp:          Specific impulse [s].
        g0:           Standard gravity [m/s²].

    Returns:
        Remaining Δv capability [m/s]. Returns 0 if no propellant remains.
    """
    if mass_current <= mass_dry or mass_dry <= 0:
        return 0.0

    exhaust_velocity = isp * g0
    return exhaust_velocity * math.log(mass_current / mass_dry)


def compute_mass_flow_rate(
    thrust_force: float,
    isp: float = DPS_ISP,
    g0: float = G0,
) -> float:
    """Compute propellant mass flow rate at a given thrust level.

    ṁ = F / (Isp × g0) = F / v_e

    This is always positive — it represents the RATE of mass consumption.
    The derivative function applies the negative sign.

    At full thrust:  ṁ = 45,040 / (311 × 9.80665) = 14.77 kg/s
    At 10% throttle: ṁ = 4,504  / (311 × 9.80665) =  1.48 kg/s

    Args:
        thrust_force: Current thrust [N].
        isp:          Specific impulse [s].
        g0:           Standard gravity [m/s²].

    Returns:
        Mass flow rate [kg/s]. Always ≥ 0.
    """
    if thrust_force <= 0.0 or isp <= 0.0:
        return 0.0
    return thrust_force / (isp * g0)


# ============================================================================
# PHASE 4 FUNCTIONS — INVERSE-SQUARE GRAVITY & ORBITAL MECHANICS
# ============================================================================
#
# WHY CONSTANT GRAVITY STOPS WORKING AT ORBITAL ALTITUDES
# ────────────────────────────────────────────────────────
# At the lunar surface, g ≈ 1.625 m/s².
# At 15.2 km altitude (PDI), g = μ/(R+h)² ≈ 1.6156 m/s².
# That's only 0.6% less — seems negligible.
#
# But over a 12-minute descent covering 50+ km of downrange distance,
# the gravity gradient across the trajectory accumulates. More importantly,
# ORBITAL MECHANICS requires the correct gravitational parameter to
# compute orbital velocity, period, and the vis-viva equation.
#
# If we used constant gravity, the orbital velocity at 15.2 km would be
# wrong, and the braking burn would leave residual velocity. Inverse-square
# gravity is what makes orbits work — it's Newton's insight.
# ============================================================================


def gravity_at_altitude(altitude: float) -> float:
    """Compute gravitational acceleration at a given altitude using inverse-square law.

    INVERSE-SQUARE GRAVITY
    ──────────────────────
    g(h) = μ / (R + h)²

    where:
        μ  = G × M_moon = 4.9028695 × 10¹² m³/s²  (gravitational parameter)
        R  = 1,737,400 m  (lunar mean radius)
        h  = altitude above mean surface [m]

    This is Newton's law of universal gravitation applied to a point mass
    at distance r = R + h from the Moon's centre. It gives the free-fall
    acceleration that any object experiences at that distance.

    COMPARISON:
        h = 0 m (surface):     g = 1.6242 m/s²
        h = 15,240 m (PDI):    g = 1.6156 m/s²  (0.5% less)
        h = 110,000 m (apolune): g = 1.4389 m/s²  (11% less)

    NOTE: The surface value from μ/R² is 1.6242, slightly different from
    the commonly cited 1.625 m/s². The 1.625 value accounts for the Moon's
    oblateness and local mascon variations. We use μ/R² consistently for
    inverse-square calculations.

    Args:
        altitude: Height above mean lunar surface [m]. Must be ≥ 0.

    Returns:
        Gravitational acceleration [m/s²], always positive.
    """
    from constants import MU_MOON, R_MOON
    r = R_MOON + max(0.0, altitude)
    return MU_MOON / (r * r)


def altitude_from_position(x: float, y: float) -> float:
    """True altitude above the lunar surface for a 2D Cartesian position.

    In the flat-world model of Phases 1-3, `y` IS the altitude. Once
    central gravity is enabled the vehicle travels around a curved body,
    so altitude must be measured radially from the Moon's centre:

        altitude = sqrt(x² + (R_MOON + y)²) − R_MOON

    Origin convention: (0, 0) is the surface point directly beneath the
    starting position, so at x = 0 this reduces to `y` exactly and stays
    backward compatible with the earlier phases.

    Args:
        x: Downrange Cartesian coordinate [m].
        y: Vertical Cartesian coordinate [m].

    Returns:
        Altitude above the mean lunar surface [m]. May be negative
        (below the reference sphere).
    """
    return math.sqrt(x * x + (R_MOON + y) ** 2) - R_MOON


def orbital_velocity(altitude: float) -> float:
    """Compute circular orbital velocity at a given altitude.

    v_circ = sqrt(μ / r)

    This is the velocity needed to maintain a circular orbit at altitude h.
    If you're moving slower, you fall. Faster, you rise. This is the
    fundamental relationship between velocity and orbital altitude.

    At PDI altitude (15.2 km):  v = sqrt(4.9028695e12 / 1752640) ≈ 1673 m/s
    At 110 km (apolune):        v = sqrt(4.9028695e12 / 1847400) ≈ 1629 m/s

    Args:
        altitude: Height above mean lunar surface [m].

    Returns:
        Circular orbital velocity [m/s].
    """
    from constants import MU_MOON, R_MOON
    r = R_MOON + altitude
    return math.sqrt(MU_MOON / r)


def vis_viva_velocity(altitude: float, semi_major_axis: float) -> float:
    """Compute orbital velocity at a given altitude using the vis-viva equation.

    THE VIS-VIVA EQUATION — conservation of orbital energy
    ──────────────────────────────────────────────────────
    v² = μ × (2/r − 1/a)

    where:
        r = R + h     (current distance from Moon centre)
        a             (semi-major axis of the orbit)

    This is the most important equation in orbital mechanics. It relates
    velocity to position for ANY point on ANY orbit. It comes directly
    from conservation of total energy:

        E = ½v² − μ/r = −μ/(2a)    (constant along the orbit)

    For the Apollo 11 descent orbit (110 km × 15.2 km):
        r_apo = R + 110,000 = 1,847,400 m
        r_peri = R + 15,240 = 1,752,640 m
        a = (r_apo + r_peri) / 2 = 1,800,020 m
        v_peri = sqrt(μ × (2/r_peri − 1/a)) ≈ 1,695 m/s

    This matches the ~1,690 m/s reference for PDI.

    Args:
        altitude: Height above surface [m].
        semi_major_axis: Orbit semi-major axis [m].

    Returns:
        Orbital velocity [m/s].
    """
    from constants import MU_MOON, R_MOON
    r = R_MOON + altitude
    return math.sqrt(MU_MOON * (2.0 / r - 1.0 / semi_major_axis))


def compute_orbital_elements(altitude: float, vx: float, vy: float) -> dict:
    """Compute basic orbital elements from current state.

    Uses the vis-viva equation in reverse: given position and velocity,
    find the orbit's semi-major axis, then compute apoapsis and periapsis.

    Energy: E = ½v² − μ/r = −μ/(2a)
    So: a = −μ / (2E)

    Then:
        r_apo = a × (1 + e)
        r_peri = a × (1 − e)

    where e is eccentricity, computed from angular momentum.

    Args:
        altitude: Height above surface [m].
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].

    Returns:
        Dict with keys: 'semi_major_axis', 'eccentricity', 'apoapsis_alt',
        'periapsis_alt', 'period', 'specific_energy', 'velocity'.
    """
    from constants import MU_MOON, R_MOON
    r = R_MOON + altitude
    v = math.sqrt(vx * vx + vy * vy)

    # Specific orbital energy
    energy = 0.5 * v * v - MU_MOON / r

    if abs(energy) < 1e-6:
        # Parabolic — edge case
        return {
            'semi_major_axis': float('inf'),
            'eccentricity': 1.0,
            'apoapsis_alt': float('inf'),
            'periapsis_alt': altitude,
            'period': float('inf'),
            'specific_energy': energy,
            'velocity': v,
        }

    # Semi-major axis
    a = -MU_MOON / (2.0 * energy)

    # Specific angular momentum (for 2D: h = r × v_tangential)
    # In our flat-surface approximation, vx is tangential and vy is radial
    h = r * vx  # angular momentum magnitude

    # Eccentricity from e = sqrt(1 + 2Eh²/μ²)
    e_squared = 1.0 + (2.0 * energy * h * h) / (MU_MOON * MU_MOON)
    e = math.sqrt(max(0.0, e_squared))

    # Apoapsis and periapsis
    if e < 1.0 and a > 0:
        r_apo = a * (1.0 + e)
        r_peri = a * (1.0 - e)
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_MOON)
    else:
        r_apo = float('inf')
        r_peri = a * (1.0 - e) if a > 0 else 0.0
        period = float('inf')

    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'apoapsis_alt': r_apo - R_MOON,
        'periapsis_alt': r_peri - R_MOON,
        'period': period,
        'specific_energy': energy,
        'velocity': v,
    }


def check_landing(altitude: float, velocity: float) -> str:
    """Determine flight status based on current state.

    Landing criteria (Phase 1/2 — vertical only):
        • altitude > 0  →  still flying
        • altitude ≤ 0 AND |velocity| ≤ 3.0 m/s  →  soft landing
        • altitude ≤ 0 AND |velocity| > 3.0 m/s   →  crash

    Args:
        altitude: Height above surface [m].
        velocity: Vertical velocity [m/s].

    Returns:
        One of "flying", "landed", or "crashed".
    """
    from constants import MAX_VERTICAL_SPEED

    if altitude > 0:
        return "flying"

    descent_rate = abs(velocity)
    if descent_rate <= MAX_VERTICAL_SPEED:
        return "landed"
    else:
        return "crashed"


# ============================================================================
# PHASE 3 FUNCTIONS — TWO-DIMENSIONAL DYNAMICS
# ============================================================================
#
# THE BIG CHANGE: THRUST VECTORING
# ─────────────────────────────────
# In Phases 1–2, thrust always pointed straight up (opposing gravity).
# Now thrust acts along the vehicle's BODY AXIS. If the LM is tilted by
# angle θ from vertical:
#
#     F_horizontal = F × sin(θ)     ← pushes sideways
#     F_vertical   = F × cos(θ)     ← fights gravity
#
# This has profound consequences:
#
# 1. At θ = 0° (upright): 100% of thrust fights gravity. Same as Phase 2.
# 2. At θ = 30°: cos(30°) = 86.6% fights gravity, sin(30°) = 50% pushes
#    sideways. You've lost 13% of your gravity authority.
# 3. At θ = 90°: cos(90°) = 0% fights gravity. You're thrusting entirely
#    sideways. You WILL crash unless you fix your attitude.
#
# This is why Apollo astronauts trained for hundreds of hours on the LLTV
# (Lunar Landing Training Vehicle). The coupling between attitude control
# and trajectory control makes this a genuinely difficult piloting problem.
#
# CONVENTION:
#     θ = 0:   vehicle upright, thrust pointing up
#     θ > 0:   tilted clockwise (to the right)
#     θ < 0:   tilted counterclockwise (to the left)
#     ω > 0:   rotating clockwise
#     Gravity: always in the -y direction
# ============================================================================


@dataclass
class SimState2D:
    """Complete 2D simulation state at one instant in time.

    The state vector has 7 components:
        [x, y, vx, vy, θ, ω, m]

    Coordinate system:
        x:  horizontal position, positive to the right
        y:  altitude, positive up (y=0 is the surface)
        vx: horizontal velocity, positive to the right
        vy: vertical velocity, positive up
        θ:  vehicle angle from vertical, positive clockwise [rad]
        ω:  angular velocity, positive clockwise [rad/s]
        m:  total mass [kg]

    Attributes:
        x:         Horizontal position [m]. Landing site is at x=0.
        y:         Altitude above surface [m].
        vx:        Horizontal velocity [m/s].
        vy:        Vertical velocity [m/s].
        theta:     Vehicle tilt angle from vertical [rad]. 0 = upright.
        omega:     Angular velocity [rad/s]. Positive = clockwise.
        mass:      Total vehicle mass [kg].
        time:      Mission elapsed time [s].
        thrust_on: Whether the DPS engine is firing.
        throttle:  DPS throttle setting [0.0–1.0].
        rcs_cmd:   RCS attitude command: -1 (left), 0 (off), +1 (right).
    """
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    mass: float
    time: float = 0.0
    thrust_on: bool = False
    throttle: float = 0.0
    rcs_cmd: int = 0


def compute_moment_of_inertia(mass: float) -> float:
    """Compute the LM's moment of inertia at a given total mass.

    WHY MOI CHANGES WITH MASS
    ─────────────────────────
    The LM's propellant is stored in four tanks arranged around the
    descent engine, offset from the centre of gravity. As propellant
    burns off, the mass far from the CG decreases, reducing the MOI.

    This means the vehicle becomes EASIER to rotate as it empties —
    the same RCS torque produces more angular acceleration. Combined
    with thrust acceleration also increasing (F/m), the LM becomes
    doubly "twitchy" near empty: it both accelerates faster linearly
    AND rotates faster angularly.

    We linearly interpolate MOI between full and empty values:
        I(m) = I_empty + (I_full - I_empty) × (m - m_dry) / m_prop

    Args:
        mass: Current total vehicle mass [kg].

    Returns:
        Moment of inertia [kg·m²].
    """
    from constants import (
        LM_MOMENT_OF_INERTIA_FULL,
        LM_MOMENT_OF_INERTIA_EMPTY,
        LM_MASS_PDI,
        PROPELLANT_MASS,
    )
    dry_mass = LM_MASS_PDI - PROPELLANT_MASS
    if PROPELLANT_MASS <= 0:
        return LM_MOMENT_OF_INERTIA_EMPTY

    # Linear interpolation: full mass → full MOI, dry mass → empty MOI
    frac = max(0.0, min(1.0, (mass - dry_mass) / PROPELLANT_MASS))
    return LM_MOMENT_OF_INERTIA_EMPTY + (
        LM_MOMENT_OF_INERTIA_FULL - LM_MOMENT_OF_INERTIA_EMPTY
    ) * frac


def derivatives_2d(
    x: float,
    y: float,
    vx: float,
    vy: float,
    theta: float,
    omega: float,
    mass: float,
    thrust_force: float,
    gravity: float,
    exhaust_velocity: float,
    rcs_torque: float,
    moment_of_inertia: float,
    central_gravity: bool = False,
) -> np.ndarray:
    """Compute time-derivatives for 2D variable-mass dynamics.

    THE 7-COMPONENT STATE VECTOR
    ────────────────────────────
    y_state = [x, y, vx, vy, θ, ω, m]

    Derivatives:
        dx/dt  = vx                                    ... horizontal motion
        dy/dt  = vy                                    ... vertical motion
        dvx/dt = (F/m) × sin(θ)                       ... thrust horizontal component
        dvy/dt = (F/m) × cos(θ) − g                   ... thrust vertical minus gravity
        dθ/dt  = ω                                     ... angular velocity
        dω/dt  = τ_rcs / I                             ... angular acceleration from RCS
        dm/dt  = −F / v_e                              ... propellant consumption

    THRUST DECOMPOSITION
    ────────────────────
    The engine nozzle points "down" relative to the vehicle body. If the
    vehicle is tilted by angle θ from vertical:

        F_x = F × sin(θ)    →  horizontal thrust component
        F_y = F × cos(θ)    →  vertical thrust component

    When θ = 0 (upright): sin=0, cos=1 → all thrust vertical (Phase 2 case)
    When θ = π/2 (sideways): sin=1, cos=0 → all thrust horizontal → crash

    ANGULAR DYNAMICS
    ────────────────
    τ = I × α  →  α = τ / I

    RCS torque is treated as an external input. In reality, the RCS jets
    fire in short pulses; here we model a continuous torque when the
    pilot commands rotation. This is a simplification but captures the
    correct angular dynamics.

    Args:
        x, y:              Position [m]. x not used in derivatives (no drag/terrain coupling).
        vx, vy:            Velocities [m/s].
        theta:             Vehicle tilt from vertical [rad].
        omega:             Angular velocity [rad/s].
        mass:              Total mass [kg].
        thrust_force:      Engine thrust [N].
        gravity:           Gravitational acceleration [m/s²], positive value.
        exhaust_velocity:  v_e = Isp × g0 [m/s].
        rcs_torque:        Net RCS torque [N·m]. Positive = clockwise.
        moment_of_inertia: Vehicle MOI about pitch axis [kg·m²].

    Returns:
        np.ndarray of shape (7,): [dx, dy, dvx, dvy, dθ, dω, dm].
    """
    dx_dt = vx
    dy_dt = vy

    # ── GRAVITY VECTOR ──────────────────────────────────────────────
    # central_gravity=False (Phases 1-3): flat-world approximation.
    #     Gravity always points in -y. Valid only for short, local
    #     trajectories where the curvature of the Moon is negligible.
    #
    # central_gravity=True (Phase 4+): gravity points toward the centre
    #     of the Moon. This is REQUIRED for orbital mechanics.
    #
    #         r_vec = (x, R_MOON + y)          position from Moon's centre
    #         a_grav = -μ · r_vec / |r_vec|³
    #
    #     Without this, horizontal velocity never curves the trajectory
    #     around the body, so a circular orbit is impossible — the
    #     vehicle simply falls while translating sideways.
    if central_gravity:
        rx = x
        ry = R_MOON + y
        r = math.sqrt(rx * rx + ry * ry)
        if r > 0.0:
            g_mag = MU_MOON / (r * r)
            gx = -g_mag * rx / r
            gy = -g_mag * ry / r
        else:
            gx = gy = 0.0
    else:
        gx = 0.0
        gy = -gravity

    if mass > 0.0 and thrust_force > 0.0:
        thrust_accel = thrust_force / mass
        # Decompose thrust along body axis into world-frame components
        dvx_dt = thrust_accel * math.sin(theta) + gx
        dvy_dt = thrust_accel * math.cos(theta) + gy
        dm_dt = -thrust_force / exhaust_velocity
    else:
        dvx_dt = gx
        dvy_dt = gy
        dm_dt = 0.0

    dtheta_dt = omega

    # Angular acceleration from RCS
    if moment_of_inertia > 0.0:
        domega_dt = rcs_torque / moment_of_inertia
    else:
        domega_dt = 0.0

    return np.array(
        [dx_dt, dy_dt, dvx_dt, dvy_dt, dtheta_dt, domega_dt, dm_dt],
        dtype=np.float64,
    )


def derivatives_2d_curvilinear(
    x: float,
    y: float,
    vx: float,
    vy: float,
    theta: float,
    omega: float,
    mass: float,
    thrust_force: float,
    exhaust_velocity: float,
    rcs_torque: float,
    moment_of_inertia: float,
) -> np.ndarray:
    """Curvilinear ("round Moon in a flat frame") derivatives.

    THE PROBLEM THIS SOLVES
    ───────────────────────
    The Cartesian central-gravity fix is physically correct, but the
    whole game — terrain, radar, display, landing checks — is written
    in a frame where y IS altitude and x IS downrange. Over a 480 km
    descent the Moon curves ~15°, so world-Cartesian coordinates and
    the game's assumptions drift apart.

    The standard aerospace answer: work in the LOCAL frame and move the
    curvature into the equations of motion. Define:

        x  = downrange distance along the SURFACE  [m]
        y  = altitude above the surface            [m]
        vx = tangential (horizontal) velocity      [m/s]
        vy = radial (vertical) velocity            [m/s]
        θ  = tilt from LOCAL vertical              [rad]
        r  = R_MOON + y

    Polar-coordinate kinematics then give:

        dx/dt  = vx · R/r          surface ground-track rate
        dy/dt  = vy
        dvx/dt = (F/m)·sinθ − vx·vy / r          (Coriolis-type term)
        dvy/dt = (F/m)·cosθ − μ/r² + vx²/r       (CENTRIFUGAL term)
        dθ/dt  = ω − vx/r          local vertical itself rotates
        dω/dt  = τ/I
        dm/dt  = −F/v_e

    The vx²/r term is everything: at orbital velocity it exactly
    cancels gravity (that IS what an orbit is), so a coasting vehicle
    at 1,672.5 m/s holds altitude forever — in the game's own frame.
    This is mathematically identical to the Cartesian central-gravity
    model (tests prove the equivalence), with zero changes needed in
    terrain, radar, display, or landing logic.
    """
    r = R_MOON + y
    dx_dt = vx * R_MOON / r
    dy_dt = vy

    g = MU_MOON / (r * r)

    if mass > 0.0 and thrust_force > 0.0:
        ta = thrust_force / mass
        ax_thrust = ta * math.sin(theta)
        ay_thrust = ta * math.cos(theta)
        dm_dt = -thrust_force / exhaust_velocity
    else:
        ax_thrust = 0.0
        ay_thrust = 0.0
        dm_dt = 0.0

    dvx_dt = ax_thrust - vx * vy / r
    dvy_dt = ay_thrust - g + vx * vx / r

    dtheta_dt = omega - vx / r
    domega_dt = rcs_torque / moment_of_inertia if moment_of_inertia > 0 else 0.0

    return np.array(
        [dx_dt, dy_dt, dvx_dt, dvy_dt, dtheta_dt, domega_dt, dm_dt],
        dtype=np.float64,
    )


def rk4_step_2d(
    x: float,
    y: float,
    vx: float,
    vy: float,
    theta: float,
    omega: float,
    mass: float,
    thrust_force: float,
    gravity: float,
    exhaust_velocity: float,
    rcs_torque: float,
    dt: float,
    dry_mass: float = 0.0,
    use_inverse_square: bool = False,
    central_gravity: bool = False,
    curvilinear: bool = False,
) -> tuple[float, float, float, float, float, float, float]:
    """Advance the 7-component state by one RK4 step.

    Same approach as Phase 2's rk4_step_with_mass, but now the state
    includes position, velocity, angle, angular velocity, and mass.

    The moment of inertia is recomputed at each RK4 sub-step because
    it depends on mass, which changes during the step. This correctly
    captures the coupling: as mass decreases, rotation response increases.

    PHASE 4 ADDITION: INVERSE-SQUARE GRAVITY
    ─────────────────────────────────────────
    When use_inverse_square=True, gravity is recomputed at each RK4
    sub-step from the current altitude: g = μ/(R+y)².

    This matters because altitude changes during each step, and at
    orbital altitudes the gravity gradient is meaningful. Over a single
    0.01s step the change is tiny, but it accumulates correctly over
    the 756-second descent.

    When use_inverse_square=False (default), the constant `gravity`
    parameter is used — backward compatible with Phases 1–3.

    Args:
        x, y:              Position [m]. y is altitude above surface.
        vx, vy:            Velocities [m/s].
        theta:             Vehicle tilt [rad].
        omega:             Angular velocity [rad/s].
        mass:              Total mass [kg].
        thrust_force:      Engine thrust [N].
        gravity:           Gravitational acceleration [m/s²]. Used when
                           use_inverse_square=False.
        exhaust_velocity:  v_e = Isp × g0 [m/s].
        rcs_torque:        RCS torque [N·m].
        dt:                Timestep [s].
        dry_mass:          Minimum mass [kg].
        use_inverse_square: If True, compute gravity from altitude at
                           each sub-step using μ/(R+y)².

    Returns:
        Tuple (x, y, vx, vy, theta, omega, mass) — new state.
    """
    state = np.array([x, y, vx, vy, theta, omega, mass], dtype=np.float64)

    def _safe_thrust(m: float) -> float:
        if m <= dry_mass or thrust_force <= 0.0:
            return 0.0
        return thrust_force

    def _derivs(s: np.ndarray) -> np.ndarray:
        """Evaluate derivatives at arbitrary sub-step state."""
        s_mass = max(s[6], dry_mass)
        moi = compute_moment_of_inertia(s_mass)
        if curvilinear:
            return derivatives_2d_curvilinear(
                x=s[0], y=s[1], vx=s[2], vy=s[3],
                theta=s[4], omega=s[5], mass=s_mass,
                thrust_force=_safe_thrust(s_mass),
                exhaust_velocity=exhaust_velocity,
                rcs_torque=rcs_torque,
                moment_of_inertia=moi,
            )
        # Phase 4: recompute gravity from altitude if inverse-square enabled
        g = gravity_at_altitude(s[1]) if use_inverse_square else gravity
        return derivatives_2d(
            x=s[0], y=s[1], vx=s[2], vy=s[3],
            theta=s[4], omega=s[5], mass=s_mass,
            thrust_force=_safe_thrust(s_mass),
            gravity=g,
            exhaust_velocity=exhaust_velocity,
            rcs_torque=rcs_torque,
            moment_of_inertia=moi,
            central_gravity=central_gravity,
        )

    # Standard RK4
    k1 = _derivs(state)

    s2 = state + 0.5 * dt * k1
    s2[6] = max(s2[6], dry_mass)
    k2 = _derivs(s2)

    s3 = state + 0.5 * dt * k2
    s3[6] = max(s3[6], dry_mass)
    k3 = _derivs(s3)

    s4 = state + dt * k3
    s4[6] = max(s4[6], dry_mass)
    k4 = _derivs(s4)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Clamp mass
    new_state[6] = max(new_state[6], dry_mass)

    return (
        float(new_state[0]),  # x
        float(new_state[1]),  # y
        float(new_state[2]),  # vx
        float(new_state[3]),  # vy
        float(new_state[4]),  # theta
        float(new_state[5]),  # omega
        float(new_state[6]),  # mass
    )


def check_landing_2d(
    y: float,
    vx: float,
    vy: float,
    theta: float,
    propellant: float,
    terrain_height: float = 0.0,
) -> tuple[str, list[str]]:
    """Determine flight status with full 2D landing criteria.

    ALL THREE LIMITS — from the Apollo LM structural specifications:
        1. Vertical descent rate ≤ 3.0 m/s
        2. Horizontal velocity ≤ 1.2 m/s
        3. Vehicle tilt ≤ 12° (0.2094 rad)
        4. Propellant > 0 kg

    If ANY limit is exceeded, the landing gear collapses → crash.
    The function returns which specific limits were violated, so the
    pilot can learn from the failure.

    Phase 6 addition: terrain_height parameter.
    The vehicle contacts the surface when y <= terrain_height,
    not when y <= 0 (which assumed flat terrain at the reference level).

    Args:
        y:              Altitude above reference [m].
        vx:             Horizontal velocity [m/s].
        vy:             Vertical velocity [m/s].
        theta:          Vehicle tilt from vertical [rad].
        propellant:     Remaining propellant [kg].
        terrain_height: Height of terrain at current x position [m].

    Returns:
        Tuple of (status, violations):
            status: "flying", "landed", or "crashed"
            violations: list of human-readable violation descriptions
    """
    from constants import MAX_VERTICAL_SPEED, MAX_HORIZONTAL_SPEED, MAX_TILT_DEGREES

    if y > terrain_height:
        return "flying", []

    violations = []
    descent_rate = abs(vy)
    horiz_speed = abs(vx)
    tilt_degrees = abs(math.degrees(theta))

    if descent_rate > MAX_VERTICAL_SPEED:
        violations.append(
            f"Vertical: {descent_rate:.1f} m/s > {MAX_VERTICAL_SPEED:.1f} limit"
        )
    if horiz_speed > MAX_HORIZONTAL_SPEED:
        violations.append(
            f"Horizontal: {horiz_speed:.1f} m/s > {MAX_HORIZONTAL_SPEED:.1f} limit"
        )
    if tilt_degrees > MAX_TILT_DEGREES:
        violations.append(
            f"Tilt: {tilt_degrees:.1f}° > {MAX_TILT_DEGREES:.1f}° limit"
        )
    if propellant <= 0.0:
        violations.append("No propellant remaining")

    if violations:
        return "crashed", violations
    else:
        return "landed", []


