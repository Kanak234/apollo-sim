"""
Physical constants for the Apollo Lunar Module descent simulator.

All values are in SI units (meters, kilograms, seconds, Newtons).
Sources are cited inline. These are real Apollo 11 / LM-5 "Eagle" parameters.

Reference documents:
    - Apollo 11 Mission Report (NASA SP-238)
    - Apollo Operations Handbook, Lunar Module (LMA790-3-LM)
    - TRW Descent Propulsion System specification
    - NASA Moon Fact Sheet (NSSDCA)
"""

# ============================================================================
# MOON
# ============================================================================

# Lunar surface gravitational acceleration.
# Source: NASA Moon Fact Sheet (https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html)
# This is the value at the mean surface. It varies slightly with location
# (mascons, highlands vs. maria) but 1.625 is the standard reference.
G_MOON: float = 1.625  # m/s²

# Lunar mean radius.
# Source: IAU 2015 recommended value.
# Used with MU_MOON for inverse-square gravity (Phase 4+).
R_MOON: float = 1_737_400.0  # m

# Lunar standard gravitational parameter (μ = G·M_moon).
# Source: DE430 planetary ephemeris (JPL).
# This is more accurate than computing G·M separately because μ is measured
# directly from orbital tracking. Individual G and M have larger uncertainties,
# but their product is known to ~10 significant figures.
MU_MOON: float = 4.9028695e12  # m³/s²


# ============================================================================
# EARTH REFERENCE (used in Isp definition only)
# ============================================================================

# Earth standard gravitational acceleration.
#
# WHY IS EARTH'S GRAVITY IN A LUNAR SIMULATOR?
# ─────────────────────────────────────────────
# Specific impulse (Isp) was historically defined as:
#     Isp = thrust / (mass_flow_rate × g0)
#
# where g0 = 9.80665 m/s² is Earth's standard gravity. This is a DEFINITION
# CONSTANT, not the local gravity. It appears because Isp was originally
# expressed as "seconds of thrust per unit weight of propellant consumed",
# with weight measured on Earth.
#
# The physically meaningful quantity is exhaust velocity:
#     v_e = Isp × g0
#
# For the DPS engine: v_e = 311 × 9.80665 ≈ 3,050 m/s
#
# The g0 cancels out when you compute fuel flow from thrust:
#     ṁ = F / (Isp × g0) = F / v_e
#
# So g0 is just a unit-conversion artifact. It has the same value whether
# the rocket is on Earth, the Moon, or in deep space.
G0: float = 9.80665  # m/s²


# ============================================================================
# LUNAR MODULE (LM-5 "Eagle")
# ============================================================================

# Total LM mass at Powered Descent Initiation (PDI).
# Source: Apollo 11 Mission Report, Table 5-I.
# This includes the descent stage, ascent stage, crew, consumables, and
# full descent propellant. The descent stage alone was ~10,334 kg; the
# ascent stage ~4,700 kg. After PDI, mass decreases as propellant burns.
LM_MASS_PDI: float = 15_100.0  # kg

# Descent stage usable propellant (Aerozine 50 fuel + N2O4 oxidizer).
# Source: Apollo 11 Mission Report, Section 5.
# "Usable" excludes trapped residuals and loading uncertainty.
# The descent stage carried about 8,200 kg that could actually be burned.
PROPELLANT_MASS: float = 8_200.0  # kg

# Descent Propulsion System (DPS) maximum thrust (vacuum).
# Source: TRW DPS specification, Apollo Operations Handbook LM.
# The DPS was a pressure-fed, hypergolic engine — Aerozine 50 and N2O4
# ignite on contact, no spark needed. This makes it extremely reliable
# (critical for a vehicle with no abort-to-runway option).
DPS_MAX_THRUST: float = 45_040.0  # N

# DPS throttle range.
# The DPS could throttle continuously from 10% to about 60% of max thrust.
# Between 60% and 100%, there was a "throttle bucket" — a forbidden zone
# where combustion became unstable. The astronaut either stayed in the
# 10-60% range for fine control, or jumped to fixed full thrust (FTP).
#
# During P63 braking, the engine ran at FTP (100%).
# During P66 manual landing, the pilot used the 10-60% range.
DPS_THROTTLE_MIN: float = 0.10           # 10% — minimum continuous
DPS_THROTTLE_MAX_CONTINUOUS: float = 0.60 # 60% — maximum continuous
DPS_THROTTLE_FULL: float = 1.00          # 100% — fixed full thrust

# DPS specific impulse (vacuum).
# Source: TRW DPS specification.
# Isp = 311 s means one kilogram of propellant produces 311 × 9.80665 ≈ 3,050 N·s
# of impulse. This is moderate by chemical rocket standards (the Space Shuttle
# Main Engine achieved ~452 s, but it used LH2/LOX which can't be stored long-term).
DPS_ISP: float = 311.0  # seconds

# Reaction Control System (RCS) — four quads of four thrusters each.
# Source: Apollo Operations Handbook LM, RCS section.
# Each quad produces ~445 N. RCS is used for attitude control (rotation)
# and small translations. Not used for main deceleration.
RCS_THRUST_PER_QUAD: float = 445.0  # N


# ============================================================================
# DESCENT ORBIT
# ============================================================================

# Perilune altitude at Powered Descent Initiation (above mean lunar surface).
# Source: Apollo 11 Mission Report, descent orbit parameters.
# After the Descent Orbit Insertion (DOI) burn, the LM was in a 110 km × 15.2 km
# orbit. PDI occurred near perilune (the lowest point).
DESCENT_ORBIT_PERILUNE: float = 15_240.0  # m

# Orbital velocity at perilune.
# Approximately computed from vis-viva: v ≈ sqrt(μ/r) for near-circular speed
# at r = R_moon + 15,240 m. The actual value depends on the full orbit, but
# ~1,690 m/s is the standard reference for the PDI state.
ORBITAL_VELOCITY_PERILUNE: float = 1_690.0  # m/s


# ============================================================================
# LANDING GEAR STRUCTURAL LIMITS
# ============================================================================
# Exceed ANY of these at touchdown and the landing gear collapses → crash.
# Source: Grumman LM structural limits, Apollo Operations Handbook.
#
# For reference, Apollo 11 actually touched down at:
#     Vertical:   ~0.5 m/s      (well within the 3.0 limit)
#     Horizontal: ~0.3 m/s      (well within the 1.2 limit)
#     Tilt:       ~4.5°         (well within the 12° limit)
# That is the standard to aim for.

MAX_VERTICAL_SPEED: float = 3.0    # m/s  — vertical descent rate at contact
MAX_HORIZONTAL_SPEED: float = 1.2  # m/s  — horizontal velocity at contact
MAX_TILT_DEGREES: float = 12.0     # deg  — vehicle tilt angle at contact


# ============================================================================
# PHASE 1 STARTING CONDITIONS
# ============================================================================
# Simplified vertical-only scenario for the minimum viable lander.
# NOT historically accurate — these are chosen to create a manageable
# challenge that teaches throttle control before orbital mechanics.
#
# At 2,000 m altitude falling at 50 m/s, you have roughly 40 seconds
# before impact (less, because you accelerate). Full thrust produces
# a net upward acceleration of ~1.36 m/s², so you need about 37 seconds
# of burn to zero your velocity. It's tight but doable.

PHASE1_START_ALT: float = 2_000.0   # m above surface
PHASE1_START_VEL: float = -50.0     # m/s (negative = falling)


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Physics timestep — 100 Hz.
# Chosen to be small enough for accurate RK4 integration of the descent
# trajectory, while large enough to not waste CPU. For constant gravity,
# RK4 is actually exact at any timestep, but smaller steps will matter
# in Phase 4 (inverse-square gravity) and Phase 6 (Kalman filter updates).
PHYSICS_DT: float = 0.01  # seconds

# Render framerate — 60 FPS.
# Decoupled from physics: the render loop draws whatever the current state
# is, while the physics loop runs its own fixed-timestep updates. This
# prevents the simulation from speeding up on fast machines or slowing
# down on slow ones.
RENDER_FPS: int = 60


# ============================================================================
# DISPLAY
# ============================================================================

WINDOW_WIDTH: int = 1024
WINDOW_HEIGHT: int = 768


# ============================================================================
# PHASE 3 — 2D DYNAMICS CONSTANTS
# ============================================================================

# RCS attitude control torque.
# The LM had four RCS quads, each with four 445 N thrusters. For pitch/roll
# control, opposing quads fire to create a torque couple. With thrusters
# approximately 2 m from the centre of gravity:
#
#   τ = F × r_arm = 445 N × 2 m = 890 N·m per pair
#
# This is the NET torque applied when the pilot commands a rotation.
# Source: Apollo Operations Handbook, LM, RCS section.
RCS_ATTITUDE_TORQUE: float = 890.0  # N·m

# LM moment of inertia about the pitch axis.
# Source: Grumman LM mass properties report.
# This is approximate and changes as propellant burns (propellant is
# stored in tanks offset from the CG). A full propellant tank adds
# more inertia; as fuel burns, the LM becomes easier to rotate.
#
# At PDI (full): ~28,000 kg·m²
# Near empty:    ~12,000 kg·m² (much twitchier rotation too!)
#
# Angular acceleration from RCS: α = τ/I = 890/28000 ≈ 0.032 rad/s²
# That's about 1.8 °/s² — it takes ~3 seconds to rotate 10°.
LM_MOMENT_OF_INERTIA_FULL: float = 28_000.0   # kg·m² at PDI
LM_MOMENT_OF_INERTIA_EMPTY: float = 12_000.0  # kg·m² at dry mass

# RCS attitude-hold (rate-nulling) autopilot gain.
#
# IMPORTANT — THIS IS NOT PHYSICAL DAMPING.
# There is no angular damping in vacuum. A tumbling spacecraft tumbles
# forever unless something torques it back. What this models is the LM's
# RCS attitude-hold autopilot, which fired thrusters to null residual
# body rates. Naming it "damping" invites the wrong mental model.
#
# It is expressed as an exponential decay RATE (per second), not as a
# per-timestep multiplier. A per-timestep factor makes the behaviour
# depend on PHYSICS_DT, which silently breaks determinism the moment the
# timestep is changed:
#
#     omega *= exp(-ATTITUDE_HOLD_RATE * dt)      <- correct, dt-independent
#     omega *= (1 - 0.02)                          <- wrong, dt-dependent
#
# 2.0203 /s reproduces the previous 2%-per-10ms behaviour exactly, so the
# handling feel is unchanged: rates decay with a time constant of ~0.5 s.
ATTITUDE_HOLD_RATE: float = 2.0203  # 1/s

# Retained for backward compatibility with older call sites.
ANGULAR_DAMPING: float = 0.02  # DEPRECATED — use ATTITUDE_HOLD_RATE


# ============================================================================
# PHASE 3 STARTING CONDITIONS
# ============================================================================
# 2D scenario: approaching the landing site from the left at moderate speed.
# The pilot must kill horizontal velocity, correct tilt, and descend —
# all while managing throttle and fuel.
#
# This is harder than Phase 2 because thrust now acts along the body axis.
# If you're tilted 30°, only cos(30°) = 87% of thrust fights gravity.

PHASE3_START_X: float = 500.0       # m — horizontal offset from landing site
PHASE3_START_Y: float = 2_000.0     # m — altitude above surface
PHASE3_START_VX: float = -10.0      # m/s — moving toward landing site (leftward)
PHASE3_START_VY: float = -50.0      # m/s — falling (same as Phase 2)
PHASE3_START_ANGLE: float = 0.05    # rad — ~3° initial tilt (not quite upright)
PHASE3_START_OMEGA: float = 0.0     # rad/s — no initial rotation

# Landing site — the target.
# x = 0 is the landing site. The vehicle starts 500 m to the right.
LANDING_SITE_X: float = 0.0         # m
LANDING_SITE_TOLERANCE: float = 50.0 # m — how close counts as "on target"


# ============================================================================
# PHASE 4 — ORBITAL MECHANICS & GUIDANCE PROGRAMS
# ============================================================================
#
# THE REAL DESCENT PROFILE
# ────────────────────────
# Apollo 11's powered descent began at 15.2 km altitude, moving nearly
# horizontally at ~1,690 m/s. The entire 12.5-minute descent was a
# controlled deceleration from orbital velocity to a near-hover at ~150 m,
# then a slow vertical descent to the surface.
#
# This is fundamentally different from Phases 1–3's "hover puzzle."
# The vehicle starts with enormous horizontal velocity and must kill it
# while simultaneously descending — a 2D trajectory optimization problem.

# ── Starting conditions at PDI ──
# The LM was in a 110 km × 15.2 km orbit after DOI.
# PDI occurs near perilune, so we start at the lowest point.
PHASE4_START_Y: float = 15_240.0    # m — perilune altitude
PHASE4_START_VX: float = 1_690.0    # m/s — orbital velocity (horizontal)
PHASE4_START_VY: float = 0.0        # m/s — at perilune, vertical rate ≈ 0
PHASE4_START_X: float = 50_000.0    # m — ~50 km downrange from landing site
PHASE4_START_ANGLE: float = 0.0     # rad — initially upright
PHASE4_START_OMEGA: float = 0.0     # rad/s

# ── Descent orbit for vis-viva ──
# 110 km × 15.2 km orbit
DESCENT_ORBIT_APOLUNE: float = 110_000.0  # m — high point of descent orbit

# ── Guidance program transition thresholds ──
# These approximate the real Apollo 11 descent timeline.
#
# P63 (Braking): Full thrust retrograde, killing orbital velocity.
#   Runs from PDI (~15.2 km, 1690 m/s) down to ~2 km altitude.
#   Consumes about 70% of descent propellant.
P63_TO_P64_ALTITUDE: float = 2_200.0    # m — transition to approach
P63_TO_P64_VELOCITY: float = 150.0      # m/s — or when horizontal vel drops below this

# P64 (Approach): Pitch over, reduced thrust, pilot sees landing site.
#   Runs from ~2 km to ~150 m.
P64_TO_P66_ALTITUDE: float = 150.0      # m — transition to terminal descent

# P66 (Rate of Descent — ROD): What Armstrong actually flew.
#   Each W/S press adjusts the target descent rate by ±0.3 m/s.
#   The computer auto-adjusts throttle to maintain the target.
#   The pilot controls attitude (horizontal position) manually.
P66_ROD_STEP: float = 0.3               # m/s per keypress
P66_DEFAULT_DESCENT_RATE: float = -0.9   # m/s — default ~3 fps descent

# P66 ROD controller gains (proportional + derivative)
# These are tuned for stable descent rate tracking.
P66_KP: float = 0.5     # Proportional gain: thrust response per m/s error
P66_KD: float = 0.3     # Derivative gain: damping for oscillation

# ── Apollo 11 descent timeline reference (MET from PDI) ──
# These are for reference and eventual Phase 7 analysis comparison.
# PDI = 000:00    Start, 15.2 km altitude
# P63 braking     000:00 → ~008:00    (8 minutes of full thrust)
# P64 approach    ~008:00 → ~010:00   (2 minutes, pitch-over)
# P66 manual      ~010:00 → ~012:30   (2.5 minutes, Armstrong flying)
# Touchdown       012:35               (12 min 35 s total)
APOLLO11_DESCENT_DURATION: float = 756.0  # seconds — total powered descent


