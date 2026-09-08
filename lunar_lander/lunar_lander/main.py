"""
Apollo Lunar Module Descent Simulator — Phase 6: Landing Radar & Terrain
══════════════════════════════════════════════════════════════════

Phase 6 adds terrain and the landing radar:
    ✓ Procedural lunar terrain with craters, ridges, boulders
    ✓ Landing radar: altitude-above-ground (AGL) measurement
    ✓ Radar lock/unlock below 12 km (40,000 ft)
    ✓ Gaussian noise on radar range and velocity
    ✓ Terrain-relative collision detection
    ✓ Terrain slope safety check (>6° = crash)
    ✓ Flat landing pad at the target site

THE LANDING RADAR — Why Apollo couldn't land without it:
    The AGC tracked position using its inertial measurement unit (IMU),
    which integrates accelerometer data. Over the ~12-minute descent,
    IMU errors could accumulate to hundreds of meters.

    The landing radar measured actual altitude above the terrain below.
    When it locked on (~12 km altitude), the AGC suddenly had a "ground
    truth" measurement. The difference between IMU and radar altitude
    (the "radar residual") told the AGC how far off its estimate was.

    During Apollo 11, the radar residual was about 800 meters — the AGC
    was that far off after 8 minutes of powered flight. Without the
    radar, they would have hit the Moon at orbital velocity.

Controls:
    W/S     ROD ±0.3 (P66) or Throttle (P67)
    A/D     RCS rotate left/right (P66/P67)
    1/2/3/4 Select P63/P64/P66/P67
    N       Cycle DSKY noun
    ENTER   Acknowledge alarm (PRO key)
    SPACE   Toggle engine on/off
    R/P/F1  Reset / Pause / Debug
    ESC     Quit
"""

from __future__ import annotations

import math
import pygame

from constants import (
    G_MOON,
    G0,
    DPS_ISP,
    DPS_MAX_THRUST,
    PROPELLANT_MASS,
    PHASE4_START_X,
    PHASE4_START_Y,
    PHASE4_START_VX,
    PHASE4_START_VY,
    PHASE4_START_ANGLE,
    PHASE4_START_OMEGA,
    LANDING_SITE_X,
    LANDING_SITE_TOLERANCE,
    ATTITUDE_HOLD_RATE,
    PHYSICS_DT,
    P66_ROD_STEP,
    R_MOON,
    DESCENT_ORBIT_PERILUNE,
    DESCENT_ORBIT_APOLUNE,
)
from physics import (
    SimState2D,
    rk4_step_2d,
    check_landing_2d,
    gravity_at_altitude,
    vis_viva_velocity,
    compute_orbital_elements,
)
from vehicle import LunarModule
from display import Display
from guidance import (
    GuidanceMode,
    GuidanceState,
    update_guidance,
)
from agc import AGC
from terrain import LunarTerrain
from radar import LandingRadar


# Guidance mode to DSKY program number
_MODE_TO_PROG = {
    GuidanceMode.P63: 63,
    GuidanceMode.P64: 64,
    GuidanceMode.P66: 66,
    GuidanceMode.P67: 67,
}


# Fuel warning thresholds
FUEL_WARNING_TIME: float = 120.0   # seconds
FUEL_CRITICAL_TIME: float = 60.0   # seconds

# Attitude rate limiter — how fast guidance can rotate the vehicle [rad/s]
ATTITUDE_RATE_LIMIT: float = 0.1   # ~5.7 °/s max guidance rotation rate


def create_initial_state(lm: LunarModule) -> SimState2D:
    """Create the starting simulation state for Phase 4 — orbital descent.

    THE PDI STATE
    ─────────────
    After Descent Orbit Insertion (DOI), the LM is in a 110 km × 15.2 km
    orbit. Powered Descent Initiation (PDI) occurs near perilune (the
    lowest point), where the vehicle is moving nearly horizontally at
    ~1,690 m/s.

    This is fundamentally different from Phases 1-3:
        Phase 3: 2,000 m altitude, 50 m/s vertical, 10 m/s horizontal
        Phase 4: 15,240 m altitude, ~0 m/s vertical, 1,690 m/s horizontal

    The challenge is completely different: you must kill 1,690 m/s of
    horizontal velocity while descending 15 km — a 2D trajectory
    optimization problem.

    Returns:
        Initial SimState2D at PDI conditions.
    """
    return SimState2D(
        x=PHASE4_START_X,
        y=PHASE4_START_Y,
        vx=PHASE4_START_VX,
        vy=PHASE4_START_VY,
        theta=PHASE4_START_ANGLE,
        omega=PHASE4_START_OMEGA,
        mass=lm.mass,
        time=0.0,
        thrust_on=False,
        throttle=0.0,
        rcs_cmd=0,
    )


def run_simulation() -> None:
    """Main simulation loop with Phase 4 orbital descent."""
    display = Display()
    lm = LunarModule()
    state = create_initial_state(lm)
    guidance = GuidanceState(mode=GuidanceMode.P63)
    agc = AGC()
    agc.set_program(_MODE_TO_PROG[GuidanceMode.P63])
    terrain = LunarTerrain(landing_site_x=LANDING_SITE_X)
    radar = LandingRadar(terrain=terrain)
    status: str = "flying"
    violations: list[str] = []
    paused: bool = False
    debug_overlay: bool = False
    physics_accumulator: float = 0.0
    fuel_warning_printed: bool = False
    fuel_critical_printed: bool = False
    initial_delta_v: float = lm.delta_v_remaining

    _print_banner(lm, state)

    running = True
    while running:
        # ────────────────────────────────────────────────────────────
        # 1. INPUT HANDLING
        # ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # ── Guidance program selection ──
                elif event.key == pygame.K_1:
                    if status == "flying":
                        guidance.mode = GuidanceMode.P63
                        guidance.auto_transition = True
                        agc.set_program(63)
                        print("  ► P63 BRAKING selected")

                elif event.key == pygame.K_2:
                    if status == "flying":
                        guidance.mode = GuidanceMode.P64
                        guidance.auto_transition = True
                        agc.set_program(64)
                        print("  ► P64 APPROACH selected")

                elif event.key == pygame.K_3:
                    if status == "flying":
                        guidance.mode = GuidanceMode.P66
                        guidance.auto_transition = False
                        agc.set_program(66)
                        print(f"  ► P66 ROD selected, target {guidance.rod_target:+.1f} m/s")

                elif event.key == pygame.K_4:
                    if status == "flying":
                        guidance.mode = GuidanceMode.P67
                        guidance.auto_transition = False
                        agc.set_program(67)
                        print("  ► P67 MANUAL selected")

                # ── Throttle / ROD controls ──
                elif event.key == pygame.K_w:
                    if status == "flying":
                        if guidance.mode == GuidanceMode.P66:
                            # In P66: W increases target (slower descent)
                            guidance.rod_target += P66_ROD_STEP
                            guidance.rod_target = min(guidance.rod_target, 2.0)
                            print(f"  ROD target: {guidance.rod_target:+.1f} m/s")
                        elif guidance.mode == GuidanceMode.P67:
                            lm.throttle_up()
                            state.throttle = lm.throttle
                            state.thrust_on = lm.engine_on

                elif event.key == pygame.K_s:
                    if status == "flying":
                        if guidance.mode == GuidanceMode.P66:
                            # In P66: S decreases target (faster descent)
                            guidance.rod_target -= P66_ROD_STEP
                            guidance.rod_target = max(guidance.rod_target, -10.0)
                            print(f"  ROD target: {guidance.rod_target:+.1f} m/s")
                        elif guidance.mode == GuidanceMode.P67:
                            lm.throttle_down()
                            state.throttle = lm.throttle
                            state.thrust_on = lm.engine_on

                elif event.key == pygame.K_SPACE:
                    if status == "flying":
                        lm.toggle_engine()
                        state.throttle = lm.throttle
                        state.thrust_on = lm.engine_on

                elif event.key == pygame.K_r:
                    lm = LunarModule()
                    state = create_initial_state(lm)
                    guidance = GuidanceState(mode=GuidanceMode.P63)
                    agc = AGC()
                    agc.set_program(63)
                    terrain = LunarTerrain(landing_site_x=LANDING_SITE_X)
                    radar = LandingRadar(terrain=terrain)
                    status = "flying"
                    violations = []
                    paused = False
                    physics_accumulator = 0.0
                    fuel_warning_printed = False
                    fuel_critical_printed = False
                    initial_delta_v = lm.delta_v_remaining
                    print("\n[RESET] Simulation restarted at PDI.\n")

                elif event.key == pygame.K_p:
                    if status == "flying":
                        paused = not paused
                        print(f"  [{'PAUSED' if paused else 'RESUMED'}]")

                elif event.key == pygame.K_F1:
                    debug_overlay = not debug_overlay

                # ── Phase 5: DSKY controls ──
                elif event.key == pygame.K_n:
                    agc.cycle_noun()
                    n = agc.registers.noun
                    print(f"  DSKY: V{agc.registers.verb:02d} N{n:02d}")

                elif event.key == pygame.K_RETURN:
                    alarm = agc.get_active_alarm()
                    if alarm:
                        agc.acknowledge_alarm()
                        print(f"  ALARM {alarm.code} acknowledged")

        # ── RCS control: held keys for continuous rotation ──
        if status == "flying" and not paused:
            keys = pygame.key.get_pressed()
            if guidance.mode in (GuidanceMode.P66, GuidanceMode.P67):
                # Manual attitude control in P66 and P67
                if keys[pygame.K_a]:
                    lm.rcs_left()
                elif keys[pygame.K_d]:
                    lm.rcs_right()
                else:
                    lm.rcs_stop()
            else:
                # In P63/P64, attitude is automatic — no manual RCS
                lm.rcs_stop()
            state.rcs_cmd = lm.rcs_command

        # ────────────────────────────────────────────────────────────
        # 2. PHYSICS UPDATE
        # ────────────────────────────────────────────────────────────
        if status == "flying" and not paused:
            frame_dt = display.clock.get_time() / 1000.0
            frame_dt = min(frame_dt, 0.1)
            physics_accumulator += frame_dt

            while physics_accumulator >= PHYSICS_DT:
                # ── Guidance update ──
                cmd = update_guidance(
                    state.x, state.y, state.vx, state.vy,
                    state.mass, guidance, PHYSICS_DT,
                )

                # ── Phase 5: Sync AGC program with guidance mode ──
                agc.set_program(_MODE_TO_PROG.get(guidance.mode, 0))

                # ── Apply guidance commands ──
                if cmd.auto_throttle and cmd.target_throttle is not None:
                    # Guidance controls throttle
                    lm.throttle = cmd.target_throttle
                    if lm.throttle > 0 and not lm.fuel_exhausted:
                        state.thrust_on = True
                    else:
                        state.thrust_on = False
                    state.throttle = lm.throttle

                if cmd.auto_attitude and cmd.target_theta is not None:
                    # Guidance controls attitude — rate-limited rotation
                    theta_error = cmd.target_theta - state.theta
                    # Rate-limit the rotation
                    max_delta = ATTITUDE_RATE_LIMIT * PHYSICS_DT
                    if abs(theta_error) > max_delta:
                        theta_error = math.copysign(max_delta, theta_error)
                    state.theta += theta_error
                    state.omega = theta_error / PHYSICS_DT  # Implied angular velocity

                # Get current thrust and RCS from vehicle model
                thrust = lm.thrust_force
                exhaust_vel = lm.exhaust_velocity
                rcs_torque = lm.rcs_torque_value

                # ── RK4 step with inverse-square gravity ──
                new_x, new_y, new_vx, new_vy, new_theta, new_omega, new_mass = (
                    # ── ROUND-MOON PHYSICS ──────────────────────
                    # curvilinear=True moves the Moon's curvature into
                    # the equations of motion while keeping y = altitude
                    # and x = downrange, which every other system here
                    # (terrain, HUD, guidance, landing checks) assumes.
                    #
                    # Without it the vehicle flies over a flat plane:
                    # a correct 15.24 km circular orbit at 1,672.5 m/s
                    # hit the surface in 138 s instead of coasting for
                    # 110 minutes. With it, orbits close to 0.00 m of
                    # drift over a full revolution and the ground track
                    # walks exactly one lunar circumference.
                    #
                    # Verified equivalent to Cartesian central gravity
                    # to 0.0000 m — see tests/test_orbital.py.
                    rk4_step_2d(
                        x=state.x,
                        y=state.y,
                        vx=state.vx,
                        vy=state.vy,
                        theta=state.theta,
                        omega=state.omega,
                        mass=state.mass,
                        thrust_force=thrust,
                        gravity=G_MOON,  # fallback, not used when inverse-square=True
                        exhaust_velocity=exhaust_vel,
                        rcs_torque=rcs_torque,
                        dt=PHYSICS_DT,
                        dry_mass=lm.dry_mass,
                        use_inverse_square=True,   # gravity falls off with altitude
                        curvilinear=True,          # ROUND MOON — see note below
                    )
                )

                # ── RCS attitude-hold (rate nulling) ──
                # Timestep-independent exponential decay. This represents the
                # autopilot firing RCS to null body rates — NOT physical
                # damping, which does not exist in vacuum.
                if not cmd.auto_attitude:
                    new_omega *= math.exp(-ATTITUDE_HOLD_RATE * PHYSICS_DT)

                # Update state
                state.x = new_x
                state.y = new_y
                state.vx = new_vx
                state.vy = new_vy
                state.theta = new_theta
                state.omega = new_omega
                state.mass = new_mass
                state.time += PHYSICS_DT
                physics_accumulator -= PHYSICS_DT

                # ── Sync mass back to vehicle model ──
                lm.sync_mass_from_physics(new_mass)

                if lm.fuel_exhausted:
                    state.thrust_on = False
                    state.throttle = 0.0
                    if not fuel_critical_printed:
                        print("  ⚠ FUEL DEPLETED — ENGINE CUTOFF")
                        fuel_critical_printed = True

                # ── Fuel warnings ──
                burn_time = lm.burn_time_remaining
                if (burn_time < FUEL_WARNING_TIME and lm.engine_on
                        and not fuel_warning_printed):
                    print(f"  ⚠ LOW FUEL — {burn_time:.0f}s at current throttle")
                    fuel_warning_printed = True
                if (burn_time < FUEL_CRITICAL_TIME and lm.engine_on
                        and not fuel_critical_printed):
                    print(f"  ⚠ FUEL CRITICAL — {burn_time:.0f}s remaining!")

                # ── Landing check — Phase 6: terrain-relative ──
                terrain_h = terrain.height_at(state.x)
                status, violations = check_landing_2d(
                    y=state.y,
                    vx=state.vx,
                    vy=state.vy,
                    theta=state.theta,
                    propellant=lm.propellant,
                    terrain_height=terrain_h,
                )
                if status != "flying":
                    state.y = max(terrain_h, state.y)
                    lm.set_engine(False)
                    lm.rcs_stop()
                    state.thrust_on = False
                    state.throttle = 0.0
                    state.rcs_cmd = 0
                    # Check terrain safety
                    if status == "landed" and not terrain.is_safe_landing(state.x):
                        slope_deg = abs(math.degrees(terrain.slope_at(state.x)))
                        violations.append(f"Terrain slope: {slope_deg:.1f}° (unsafe)")
                        status = "crashed"
                    _print_landing_report(state, status, violations, lm,
                                         initial_delta_v, guidance)
                    break

        # ────────────────────────────────────────────────────────────
        # 3. RENDER
        # ────────────────────────────────────────────────────────────
        fuel_warn = (
            lm.engine_on
            and not lm.fuel_exhausted
            and lm.burn_time_remaining < FUEL_WARNING_TIME
        )

        # Compute orbital elements for display
        orb = compute_orbital_elements(state.y, state.vx, state.vy)

        # ── Phase 6: Update landing radar ──
        radar_reading = radar.update(
            state.x, state.y, state.vx, state.vy, state.time,
        )
        radar_alt = radar_reading.range_m if radar_reading.range_valid else state.y
        altitude_agl = radar.altitude_agl(state.x, state.y)

        # ── Phase 5: Update AGC display data and alarms ──
        agc.update_display(
            altitude=radar_alt if radar.is_locked else state.y,
            velocity=state.vy,
            vx=state.vx,
            vy=state.vy,
            theta=state.theta,
            omega=state.omega,
            mass=state.mass,
            delta_v=lm.delta_v_remaining,
            burn_time=lm.burn_time_remaining,
            propellant=lm.propellant,
            propellant_max=PROPELLANT_MASS,
            x=state.x,
            landing_site_x=LANDING_SITE_X,
            time_elapsed=state.time,
        )

        # Maybe trigger a 1202 alarm (historically accurate!)
        if status == "flying":
            guidance_active = guidance.mode in (GuidanceMode.P63, GuidanceMode.P64)
            agc.maybe_trigger_1202(state.time, state.thrust_on, guidance_active)

        display.render(
            altitude=altitude_agl,
            velocity=state.vy,
            time_elapsed=state.time,
            thrust_on=state.thrust_on,
            status=status,
            paused=paused,
            debug=debug_overlay,
            # Phase 2 props
            throttle=lm.throttle,
            propellant=lm.propellant,
            propellant_max=PROPELLANT_MASS,
            mass=lm.mass,
            delta_v=lm.delta_v_remaining,
            burn_time=lm.burn_time_remaining,
            fuel_warning=fuel_warn,
            # Phase 3 props
            x=state.x,
            vx=state.vx,
            theta=state.theta,
            omega=state.omega,
            rcs_cmd=state.rcs_cmd,
            landing_site_x=LANDING_SITE_X,
            violations=violations,
            # Phase 5 props
            agc_registers=agc.registers,
            guidance_mode=guidance.mode.value,
            rod_target=guidance.rod_target,
            # Phase 6 props
            terrain=terrain,
            radar_locked=radar.is_locked,
            radar_alt=radar_alt,
        )

    display.cleanup()
    print("\nSimulation ended. Fly safe.\n")


def _print_banner(lm: LunarModule, state: SimState2D) -> None:
    """Print Phase 4 simulation parameters."""

    # Compute orbital velocity at perilune using vis-viva
    r_apo = R_MOON + DESCENT_ORBIT_APOLUNE
    r_peri = R_MOON + DESCENT_ORBIT_PERILUNE
    sma = (r_apo + r_peri) / 2.0
    v_peri = vis_viva_velocity(DESCENT_ORBIT_PERILUNE, sma)

    # Gravity at PDI altitude
    g_pdi = gravity_at_altitude(PHASE4_START_Y)
    g_surface = gravity_at_altitude(0.0)

    max_flow = DPS_MAX_THRUST / (DPS_ISP * G0)
    max_burn_time = lm.propellant / max_flow if max_flow > 0 else 0

    print()
    print("═" * 66)
    print("  APOLLO LM DESCENT SIMULATOR — PHASE 6")
    print("  Landing Radar & Terrain: Powered Descent from 15.2 km")
    print("═" * 66)
    print(f"  Descent orbit:        {DESCENT_ORBIT_APOLUNE/1000:.0f} km"
          f" × {DESCENT_ORBIT_PERILUNE/1000:.1f} km")
    print(f"  PDI altitude:         {state.y:>10.0f} m  ({state.y/1000:.1f} km)")
    print(f"  Orbital velocity:     {state.vx:>10.0f} m/s"
          f"  (vis-viva: {v_peri:.0f} m/s)")
    print(f"  Gravity at PDI:       {g_pdi:>10.4f} m/s²"
          f"  (surface: {g_surface:.4f})")
    print("─" * 66)
    print(f"  Vehicle mass:         {lm.mass:>10.0f} kg")
    print(f"  Propellant:           {lm.propellant:>10.0f} kg")
    print(f"  Max burn time:        {max_burn_time:>10.0f} s"
          f"  ({max_burn_time / 60:.1f} min)")
    print(f"  Δv available:         {lm.delta_v_remaining:>10.0f} m/s")
    print("─" * 66)
    print("  Phase 6 — Landing Radar:")
    print("    • Radar locks below ~12 km altitude (40,000 ft)")
    print("    • Measures altitude above actual terrain (AGL)")
    print("    • Procedural lunar terrain with craters & slopes")
    print("    • Landing pad at x=0 with safe flat zone")
    print("    • Terrain slope > 6° = crash on landing")
    print("─" * 66)
    print("  Controls: N=noun, ENTER=ack, 1-4=program, W/S=ROD, A/D=RCS")
    print("  Auto-transition: P63 → P64 at ~2.2 km  → P66 at ~150 m")
    print("═" * 66)
    print()


def _print_landing_report(
    state: SimState2D,
    status: str,
    violations: list[str],
    lm: LunarModule,
    initial_delta_v: float,
    guidance: GuidanceState,
) -> None:
    """Print detailed landing report for orbital descent."""
    descent_rate = abs(state.vy)
    horiz_speed = abs(state.vx)
    tilt_deg = abs(math.degrees(state.theta))
    delta_v_used = initial_delta_v - lm.delta_v_remaining
    fuel_used = PROPELLANT_MASS - lm.propellant
    distance = abs(state.x - LANDING_SITE_X)

    print()
    print("═" * 66)
    if status == "landed":
        if distance <= LANDING_SITE_TOLERANCE:
            print("  ★  THE EAGLE HAS LANDED  ★")
            print(f"      Tranquility Base, distance {distance:.0f} m from target")
        else:
            print("  ★  LANDED — off target  ★")
            print(f"      {distance:.0f} m from the landing site")
        print("─" * 66)
        print(f"  Vertical velocity:    {state.vy:>+10.3f} m/s  (limit 3.0)")
        print(f"  Horizontal velocity:  {state.vx:>+10.3f} m/s  (limit 1.2)")
        print(f"  Vehicle tilt:         {tilt_deg:>10.1f}°    (limit 12.0°)")
        print("─" * 66)
        # Compare to Apollo 11
        print(f"  Descent time:         {state.time:>10.1f} s"
              f"  (Apollo 11: 756 s)")
        print(f"  Δv used:              {delta_v_used:>10.0f} m/s"
              f"  (Apollo 11: ~2,050 m/s)")
        if descent_rate <= 0.5 and horiz_speed <= 0.3 and tilt_deg <= 5.0:
            print("  Rating: PERFECT — Neil Armstrong level!")
        elif descent_rate <= 1.0 and horiz_speed <= 0.6:
            print("  Rating: EXCELLENT")
        elif descent_rate <= 2.0:
            print("  Rating: GOOD — safe but rough")
        else:
            print("  Rating: MARGINAL — gear survived, barely")
    else:
        print("  ✖  VEHICLE LOST — CRASH  ✖")
        print("─" * 66)
        print("  Violations:")
        for v in violations:
            print(f"    ✖ {v}")
        print("─" * 66)
        print(f"  Impact vertical:      {state.vy:>+10.3f} m/s")
        print(f"  Impact horizontal:    {state.vx:>+10.3f} m/s")
        print(f"  Impact tilt:          {tilt_deg:>10.1f}°")

    print("─" * 66)
    print(f"  Final program:        {guidance.mode.value}")
    print(f"  Mission time:         {state.time:>10.1f} s"
          f"  ({state.time / 60:.1f} min)")
    print(f"  Fuel remaining:       {lm.propellant:>10.1f} kg"
          f"  ({lm.propellant_fraction * 100:.1f}%)")
    print(f"  Fuel used:            {fuel_used:>10.1f} kg")
    print(f"  Δv remaining:         {lm.delta_v_remaining:>10.0f} m/s")
    print("═" * 66)
    print("  Press R to restart, ESC to quit.")
    print()


if __name__ == "__main__":
    run_simulation()
