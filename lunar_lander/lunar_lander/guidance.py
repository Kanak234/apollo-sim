"""
Guidance programs for the Apollo LM descent — Phase 4.

This module implements the four descent guidance programs as they
were actually used during Apollo 11's powered descent:

    P63 — Braking Phase
    P64 — Approach Phase
    P66 — Rate of Descent (ROD) — what Armstrong actually flew
    P67 — Full Manual — direct throttle and attitude control

THE DESCENT PROFILE
───────────────────
Apollo 11's powered descent was NOT a vertical drop. It was a
12.5-minute controlled deceleration from orbital velocity (~1,690 m/s
horizontal at 15.2 km altitude) to a near-hover at ~150 m, then a
slow vertical descent to the surface.

The descent had three phases:

    P63 (Braking):   ~8 minutes, full thrust retrograde
        Kill most of the 1,690 m/s horizontal velocity.
        Consumes ~70% of descent propellant.
        Vehicle is pitched nearly horizontal, thrusting backward.

    P64 (Approach):  ~2 minutes, reduced thrust
        Pitch over to near-vertical.
        Pilot can see the landing site through the window.
        Landing Point Designator shows where the trajectory leads.

    P66 (ROD Manual): ~2.5 minutes, pilot-controlled
        Computer holds a target descent rate.
        Each pilot input adjusts the target by ±0.3 m/s (±1 fps).
        Pilot controls attitude (horizontal position) manually.
        This is how Neil Armstrong actually flew the last 150 m.

    P67 (Full Manual): not normally used
        Direct throttle and attitude control.
        Same as Phase 3 behavior — no computer assistance.

GUIDANCE PHILOSOPHY
───────────────────
The key insight: these programs don't fly the vehicle — they compute
what the vehicle SHOULD do, and the pilot or autopilot executes it.
The separation between guidance (what to do) and control (how to do it)
is one of the most important architectural patterns in flight software.

This module implements the guidance side: given the current state,
what should the throttle and attitude be?
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass

from constants import (
    MU_MOON,
    R_MOON,
    DPS_MAX_THRUST,
    DPS_MAX_THRUST,
    DPS_THROTTLE_MIN,
    DPS_THROTTLE_MAX_CONTINUOUS,
    DPS_THROTTLE_FULL,
    P63_TO_P64_ALTITUDE,
    P63_TO_P64_VELOCITY,
    P64_TO_P66_ALTITUDE,
    P66_DEFAULT_DESCENT_RATE,
    P66_KP,
    P66_KD,
)


# ── P64 approach tuning (swept; see test_descent_mission.py) ────────
_AH_CAP = 4.5        # m/s^2 — max horizontal braking authority
_SINK_HOLD = 15.0    # m/s — sink rate held while horizontal vel remains
_KP = 0.5            # sink-rate proportional gain


class GuidanceMode(Enum):
    """Descent guidance programs."""
    P63 = "P63"  # Braking Phase — full thrust retrograde
    P64 = "P64"  # Approach Phase — pitch-over, reduced thrust
    P66 = "P66"  # Rate of Descent — manual descent rate control
    P67 = "P67"  # Full Manual — direct control (Phase 3 behavior)


@dataclass
class GuidanceCommand:
    """Output from the guidance computer: what the vehicle should do.

    The guidance computer does NOT directly control the vehicle.
    It computes target values that the control system (or pilot) executes.

    Attributes:
        target_theta:    Desired vehicle angle [rad]. None = pilot controls.
        target_throttle: Desired throttle [0.0–1.0]. None = pilot controls.
        auto_attitude:   If True, guidance controls attitude automatically.
        auto_throttle:   If True, guidance controls throttle automatically.
        mode:            Current guidance program.
        status_text:     Human-readable status for the HUD.
    """
    target_theta: float | None = None
    target_throttle: float | None = None
    auto_attitude: bool = False
    auto_throttle: bool = False
    mode: GuidanceMode = GuidanceMode.P67
    status_text: str = ""


@dataclass
class GuidanceState:
    """Internal state of the guidance computer.

    Tracks the current program, P66 target descent rate, and
    auto-transition logic.

    Attributes:
        mode:               Current guidance program.
        rod_target:         P66 target descent rate [m/s]. Negative = descending.
        p63_initiated:      Whether P63 braking has been started.
        auto_transition:    Whether to automatically switch P63→P64→P66.
        previous_vy:        Previous frame's vertical velocity (for P66 derivative).
    """
    mode: GuidanceMode = GuidanceMode.P67
    rod_target: float = P66_DEFAULT_DESCENT_RATE
    p63_initiated: bool = False
    auto_transition: bool = True
    previous_vy: float = 0.0


def compute_retrograde_angle(vx: float, vy: float) -> float:
    """Compute the vehicle angle that points thrust opposite to velocity.

    RETROGRADE BURN
    ───────────────
    To decelerate (kill velocity), you thrust OPPOSITE to your velocity
    vector. If you're moving mostly horizontally (as at PDI), the vehicle
    must pitch nearly horizontal and thrust backward.

    The retrograde direction is:
        θ_retro = atan2(-vx, -vy)

    But our convention is θ = 0 = upright, positive clockwise.
    And thrust points along the vehicle's +Y body axis (up from cockpit).

    So to thrust retrograde, the vehicle body axis must point anti-velocity:
        θ = atan2(-vx, -vy)

    At PDI (vx=1690, vy≈0):
        θ = atan2(-1690, 0) = -π/2 = -90°
    → Vehicle tilted 90° to the left, thrusting backward.

    As horizontal velocity is killed and vertical velocity grows:
        θ gradually rotates toward 0 (upright).

    This is the natural pitch-over that P63 performs.

    Args:
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].

    Returns:
        Retrograde angle [rad] in our convention.
    """
    v_mag = math.sqrt(vx * vx + vy * vy)
    if v_mag < 0.1:
        return 0.0  # Near zero velocity → point up

    # Anti-velocity direction
    return math.atan2(-vx, -vy)


def compute_p63_guidance(
    vx: float,
    vy: float,
    altitude: float,
    mass: float,
) -> GuidanceCommand:
    """P63 Braking Phase guidance.

    BRAKING PHASE
    ─────────────
    Full thrust, retrograde orientation. The computer points the vehicle
    opposite to the velocity vector and fires at full power.

    This is the workhorse of the descent: it kills most of the 1,690 m/s
    horizontal velocity over about 8 minutes, consuming ~70% of the
    descent propellant.

    The vehicle starts pitched nearly horizontal (thrust backward) and
    gradually rotates upward as horizontal velocity decreases.

    Transition to P64 when:
        - Altitude drops below ~2,200 m, OR
        - Horizontal velocity drops below ~150 m/s

    Args:
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].
        altitude: Current altitude [m].
        mass: Current vehicle mass [kg].

    Returns:
        GuidanceCommand with retrograde angle and full throttle.
    """
    v_total = math.sqrt(vx * vx + vy * vy)

    # ── COORDINATED BRAKING ─────────────────────────────────────────
    # P63 is NOT pure retrograde. Pointing straight along -v leaves no
    # vertical thrust component, so the vehicle free-falls while it
    # brakes and arrives at the surface still doing hundreds of m/s.
    # (Pure-retrograde P63 crashes at ~680-1,240 m/s horizontal — this
    # is what the end-to-end mission test caught.)
    #
    # Real P63 tilts the thrust vector so ONE component holds the sink
    # rate bounded while the REST goes into braking. Engine stays at
    # full thrust; attitude does the steering. So:
    #
    #     a_up needed   = g + k·(sink_target − vy)
    #     cos(theta)    = a_up / (F_max/m)        ... vertical share
    #     the remaining sin(theta) automatically brakes
    #
    # The sink target is scheduled off time-to-go: how long the
    # remaining horizontal velocity takes to null at ~3 m/s^2 sets how
    # fast we can afford to descend, so altitude runs out just as
    # horizontal velocity does.
    r = R_MOON + max(altitude, 0.0)
    g_local = MU_MOON / (r * r)

    v_h = abs(vx)
    t_go = max(v_h / 3.0, 20.0)
    sink_target = -min(60.0, max(3.0, 0.85 * max(altitude, 0.0) / t_go))

    a_up = g_local + 0.35 * (sink_target - vy)
    a_max = DPS_MAX_THRUST / mass if mass > 0.0 else 0.0

    if a_max <= 0.0:
        theta_cmd = 0.0
    else:
        cos_theta = max(-1.0, min(1.0, a_up / a_max))
        theta_cmd = math.acos(cos_theta)
        # tilt so the horizontal component OPPOSES vx
        if vx > 0.0:
            theta_cmd = -theta_cmd

    return GuidanceCommand(
        target_theta=theta_cmd,
        target_throttle=DPS_THROTTLE_FULL,
        auto_attitude=True,
        auto_throttle=True,
        mode=GuidanceMode.P63,
        status_text=f"BRAKING  V={v_total:.0f} m/s",
    )


def compute_p64_guidance(
    vx: float,
    vy: float,
    altitude: float,
    mass: float,
) -> GuidanceCommand:
    """P64 Approach Phase guidance.

    APPROACH PHASE
    ──────────────
    After P63 has killed most horizontal velocity, P64 pitches the
    vehicle more upright so the pilot can see the landing site through
    the window.

    The thrust is reduced and the vehicle angle is computed to maintain
    a roughly constant descent profile toward the landing site.

    A simplified version: point ~30° from vertical toward the remaining
    horizontal velocity, at about 40% throttle.

    Transition to P66 when altitude drops below ~150 m.

    Args:
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].
        altitude: Current altitude [m].
        mass: Current vehicle mass [kg].

    Returns:
        GuidanceCommand with approach angle and moderate throttle.
    """
    # ── COORDINATED APPROACH ────────────────────────────────────────
    # Two bugs lived here and both were mission-fatal:
    #
    #   1. SIGN. The tilt was copysign(angle, vx), which points the
    #      horizontal thrust component the SAME way as vx — accelerating
    #      the vehicle sideways instead of braking it. Horizontal speed
    #      climbed 150 -> 190 m/s during "approach".
    #
    #   2. THROTTLE. Commanding hover + 0.05 cannot arrest a 59 m/s
    #      descent; it barely holds altitude. The vehicle flew a
    #      constant-sink-rate line straight into the surface.
    #
    # The fix is the same shape as P63: decide the acceleration vector
    # first (vertical share for sink control, horizontal share for
    # braking), then convert to attitude + throttle.
    r = R_MOON + max(altitude, 0.0)
    g = MU_MOON / (r * r)

    # Sink schedule: gentler as the ground approaches, handing P66 a
    # vehicle that is already slow.
    # Kill horizontal velocity FIRST, holding a moderate sink, then
    # descend vertically. Trying to do both at once runs out of
    # altitude before it runs out of horizontal velocity.
    if abs(vx) > 10.0:
        sink_target = -_SINK_HOLD
    else:
        sink_target = -min(_SINK_HOLD, max(1.5, altitude / 40.0))
    a_up = g + _KP * (sink_target - vy)
    a_h = min(_AH_CAP, 0.5 * abs(vx))

    a_cmd = math.hypot(a_up, a_h)
    throttle = (mass * a_cmd) / DPS_MAX_THRUST if DPS_MAX_THRUST > 0 else 0.0

    # Respect the throttle bucket: 10-60 % continuous, or fixed 100 %.
    if throttle > DPS_THROTTLE_MAX_CONTINUOUS:
        throttle = DPS_THROTTLE_FULL
    throttle = max(DPS_THROTTLE_MIN, min(DPS_THROTTLE_FULL, throttle))

    if a_cmd > 1e-6:
        theta_target = math.asin(max(-1.0, min(1.0, a_h / a_cmd)))
        if vx > 0.0:
            theta_target = -theta_target      # oppose vx, never follow it
    else:
        theta_target = 0.0

    v_total = math.sqrt(vx * vx + vy * vy)

    return GuidanceCommand(
        target_theta=theta_target,
        target_throttle=throttle,
        auto_attitude=True,
        auto_throttle=True,
        mode=GuidanceMode.P64,
        status_text=f"APPROACH  V={v_total:.0f}  ALT={altitude:.0f}",
    )


def compute_p66_guidance(
    vx: float,
    vy: float,
    altitude: float,
    mass: float,
    rod_target: float,
    previous_vy: float,
    dt: float,
    autopilot: bool = False,
) -> GuidanceCommand:
    """P66 Rate of Descent (ROD) guidance — what Armstrong actually used.

    P66 — THE MOST IMPORTANT GUIDANCE MODE
    ───────────────────────────────────────
    The computer holds a target descent rate (vertical velocity).
    The pilot adjusts this rate with discrete inputs:
        W → rod_target += 0.3 m/s (slower descent / ascend more)
        S → rod_target -= 0.3 m/s (faster descent)

    The computer auto-adjusts throttle to maintain the target rate
    using a simple proportional-derivative controller:
        throttle = hover_throttle + Kp × (vy − target) + Kd × d(vy)/dt

    Meanwhile, the pilot controls attitude (A/D keys) to manage
    horizontal position — steering toward the landing site.

    This is how Neil Armstrong actually flew the final 150 m of the
    Apollo 11 descent. He used P66 because P65 (fully automatic) was
    steering toward a boulder field, so he took manual control.

    Args:
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].
        altitude: Current altitude [m].
        mass: Current vehicle mass [kg].
        rod_target: Target descent rate [m/s], negative = descending.
        previous_vy: Vertical velocity from the last frame [m/s].
        dt: Time since last frame [s].

    Returns:
        GuidanceCommand with auto-throttle, manual attitude.
    """
    from physics import gravity_at_altitude
    g = gravity_at_altitude(altitude)

    # Hover throttle: the throttle that exactly cancels gravity at current mass
    hover_thrust = mass * g
    hover_throttle = hover_thrust / DPS_MAX_THRUST

    # PD controller: adjust throttle to drive vy toward rod_target
    # Sign convention: if vy is more negative than target (descending
    # too fast), error is POSITIVE → increase throttle.
    # Example: vy=-3.0, target=-0.9 → error = -0.9 - (-3.0) = +2.1 → more thrust
    # ── AUTOPILOT ROD SCHEDULE ──────────────────────────────────────
    # A flat -0.9 m/s from 150 m means 167 s of powered hover, which
    # the DPS cannot pay for — the end-to-end test ran the tanks dry at
    # 60 m and fell. Real P66 comes down briskly and eases off near the
    # surface. When a human is flying, their W/S inputs own this value
    # and the schedule stays out of the way.
    if autopilot:
        rod_target = -min(3.0, max(0.9, altitude / 50.0))

    error = rod_target - vy

    # ── RATE HOLD AS AN ACCELERATION COMMAND ────────────────────────
    # The original law was throttle = hover + Kp·e + Kd·de/dt, with
    # de/dt differentiated numerically at the physics timestep. That
    # derivative is just vehicle acceleration, which is itself set by
    # the throttle one step earlier — so the loop closed on its own
    # output and chattered between 10 % and 41 % every step, holding
    # -0.6 m/s when -3.0 was commanded and burning the tanks dry.
    #
    # Commanding the ACCELERATION instead removes the loop entirely:
    # ask for the vertical acceleration that drives vy to the target,
    # then solve for the throttle that produces it.
    r = R_MOON + max(altitude, 0.0)
    g_local = MU_MOON / (r * r)

    a_up = g_local + P66_KP * error
    throttle = (mass * a_up) / DPS_MAX_THRUST if DPS_MAX_THRUST > 0 else 0.0

    # Clamp to valid throttle range
    # P66 allows full throttle for emergency deceleration
    throttle = max(DPS_THROTTLE_MIN, min(DPS_THROTTLE_FULL, throttle))

    return GuidanceCommand(
        # Attitude in P66 belongs to the pilot — this is the mode
        # Armstrong hand-flew past the boulder field. Under autopilot
        # there is no pilot, so hold upright: the attitude-hold law
        # damps the residual RATE but never nulls the residual ANGLE,
        # so a few tenths of a degree left over at handoff integrates
        # into metres per second of lateral drift by touchdown.
        target_theta=0.0 if autopilot else None,
        target_throttle=throttle,
        auto_attitude=autopilot,
        auto_throttle=True,        # Computer controls throttle
        mode=GuidanceMode.P66,
        status_text=f"ROD {rod_target:+.1f} m/s  ALT={altitude:.0f}",
    )


def compute_p67_guidance() -> GuidanceCommand:
    """P67 Full Manual — no computer assistance.

    Same as Phase 3 behavior: pilot controls both throttle and attitude.
    The computer is just monitoring, not commanding.
    """
    return GuidanceCommand(
        target_theta=None,
        target_throttle=None,
        auto_attitude=False,
        auto_throttle=False,
        mode=GuidanceMode.P67,
        status_text="MANUAL",
    )


def update_guidance(
    state_x: float,
    state_y: float,
    state_vx: float,
    state_vy: float,
    state_mass: float,
    guidance_state: GuidanceState,
    dt: float,
) -> GuidanceCommand:
    """Main guidance update — dispatch to the active program.

    Also handles automatic program transitions (P63 → P64 → P66)
    based on altitude and velocity thresholds.

    Args:
        state_x, state_y: Position [m].
        state_vx, state_vy: Velocities [m/s].
        state_mass: Current total mass [kg].
        guidance_state: Mutable guidance state (modified in-place for transitions).
        dt: Time since last update [s].

    Returns:
        GuidanceCommand for the current program.
    """
    mode = guidance_state.mode

    # ── Auto-transition logic ──
    if guidance_state.auto_transition:
        if mode == GuidanceMode.P63:
            # Transition P63 → P64 when horizontal velocity is nearly
            # dead. The altitude condition is an AND, not an OR: being
            # low while still doing 1,200 m/s is not "approach", it is
            # a failed braking phase, and handing off to P64 there
            # guarantees a crash.
            if (abs(state_vx) < P63_TO_P64_VELOCITY
                    or (state_y < P63_TO_P64_ALTITUDE
                        and abs(state_vx) < 3.0 * P63_TO_P64_VELOCITY)):
                guidance_state.mode = GuidanceMode.P64
                mode = GuidanceMode.P64
                print(f"  ► P64 APPROACH — altitude {state_y:.0f} m,"
                      f" horiz vel {abs(state_vx):.0f} m/s")

        elif mode == GuidanceMode.P64:
            # Transition P64 → P66 when close to surface
            if state_y < P64_TO_P66_ALTITUDE:
                guidance_state.mode = GuidanceMode.P66
                mode = GuidanceMode.P66
                print(f"  ► P66 ROD — altitude {state_y:.0f} m,"
                      f" target rate {guidance_state.rod_target:+.1f} m/s")

    # ── Dispatch to active program ──
    if mode == GuidanceMode.P63:
        cmd = compute_p63_guidance(state_vx, state_vy, state_y, state_mass)
    elif mode == GuidanceMode.P64:
        cmd = compute_p64_guidance(state_vx, state_vy, state_y, state_mass)
    elif mode == GuidanceMode.P66:
        cmd = compute_p66_guidance(
            state_vx, state_vy, state_y, state_mass,
            guidance_state.rod_target,
            guidance_state.previous_vy,
            dt,
            autopilot=guidance_state.auto_transition,
        )
    else:  # P67
        cmd = compute_p67_guidance()

    # Update previous_vy for P66 derivative term
    guidance_state.previous_vy = state_vy

    return cmd
