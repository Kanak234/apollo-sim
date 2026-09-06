# Build Prompt II — Full Mission: Earth Launch → Moon → Mars

> **Scope warning, read this first.** This document covers Phases 8 to 19. You are
> currently on Phase 1. Do not paste this whole file into a coding assistant — it would
> produce thousands of lines of half-working code and you would learn nothing.
>
> This exists so you know where the project is going and can build the early phases with
> the right structure. Finish Phases 1–7 from the first prompt. Then come back and paste
> **the Core Physics Rules section plus exactly one phase.**
>
> Realistic estimate: Phases 1–7 is a few months of evenings. Phases 8–13 is a serious
> personal project. Phases 14–19 is dissertation-scale. That is fine — it is meant to be
> the thing you keep building through MCA.

---

## PROMPT BEGINS HERE

You are an experienced flight-dynamics and mission-design engineer. I am a BCA student
with strong Python skills and a working Apollo Lunar Module descent simulator already
built (variable-mass dynamics, hand-written RK4, DSKY interface, P63–P67 descent
programs, 1202 alarm simulation).

I am now extending it into a **full mission simulator**: launch from Earth, fly the exact
Apollo 11 profile to the Moon and back — and then a second campaign that reuses the same
engine to reach Mars using orbital propellant depots and in-situ resource production.

Explain the physics and the mission-design reasoning as you implement. I want to
understand why each manoeuvre exists, not just watch numbers change.

---

# CORE PHYSICS RULES — APPLY TO EVERY PHASE

These carry forward from the existing simulator and are non-negotiable.

**Dynamics**
- Newtonian mechanics, F = ma, variable-mass vehicles throughout.
- Propellant mass flow: `dm/dt = -F_thrust / (Isp * g0)` with `g0 = 9.80665 m/s²`.
- Delta-v accounting via Tsiolkovsky: `Δv = Isp * g0 * ln(m0/mf)`.
- Hand-written RK4 with fixed timestep, decoupled from render rate. Adaptive stepping is
  permitted from Phase 9 onward where timescales vary by orders of magnitude — if you add
  it, explain the error-control logic.

**Gravity**
- Inverse-square gravity from every relevant body: `a = -μ * r_vec / |r|³`.
- Patched-conic approximation is acceptable for interplanetary phases. State clearly in
  comments where it breaks down and what a full n-body integration would change.
- Include J2 oblateness perturbation for Earth orbits from Phase 8.

**Atmosphere (new — this did not exist in the lunar simulator)**
- Earth: exponential density model `ρ = ρ0 * exp(-h/H)` with `ρ0 = 1.225 kg/m³` and scale
  height `H ≈ 8,500 m`, valid to roughly 100 km. Use the US Standard Atmosphere table
  above that if accuracy matters.
- Mars: `ρ0 ≈ 0.020 kg/m³`, scale height `H ≈ 11,100 m`.
- Drag: `F_drag = 0.5 * ρ * v² * Cd * A`, opposing the velocity vector.
- Gravity losses and drag losses must be tracked and reported separately, because
  understanding where the delta-v actually goes is one of the main learning objectives.

**Forbidden — do not implement under any framing**
- Magnetic or "anti-gravity" propulsion. Gravity and magnetism are unrelated forces and a
  spacecraft is not ferromagnetic. Nothing pushes off a planet's magnetic field.
- Reactionless drives of any kind.
- Any propulsion that does not either expel reaction mass or transfer photon/particle
  momentum from an external source.

---

# PART A — THE FULL APOLLO 11 MISSION

Reproduce the mission exactly. Every constant and every timestamp below is real.

## Saturn V (AS-506) vehicle data

**Overall**
| Parameter | Value |
|---|---|
| Liftoff mass | ≈ 2,970,000 kg |
| Height | 110.6 m |
| Liftoff thrust-to-weight ratio | ≈ 1.14 |
| Launch site | LC-39A, Kennedy Space Center (28.608°N, 80.604°W) |
| Launch azimuth | 72.058° |
| Liftoff | 16 July 1969, 13:32:00 UTC |

**S-IC — first stage**
| Parameter | Value |
|---|---|
| Engines | 5 × F-1, RP-1 / LOX |
| Thrust (sea level, total) | ≈ 33,400 kN |
| Isp | 263 s sea level, 304 s vacuum |
| Propellant mass | ≈ 2,077,000 kg |
| Burn time | ≈ 168 s |
| Centre engine cutoff | T+135 s (to cap acceleration at 4 g) |
| Cutoff conditions | ≈ 67 km altitude, ≈ 2,760 m/s |

**S-II — second stage**
| Parameter | Value |
|---|---|
| Engines | 5 × J-2, LH2 / LOX |
| Thrust (vacuum, total) | ≈ 5,000 kN |
| Isp | ≈ 421 s |
| Propellant mass | ≈ 456,000 kg |
| Burn time | ≈ 384 s |
| Cutoff conditions | ≈ 185 km altitude, ≈ 6,900 m/s |

**S-IVB — third stage**
| Parameter | Value |
|---|---|
| Engine | 1 × J-2 (restartable) |
| Thrust (vacuum) | ≈ 1,000 kN |
| Isp | ≈ 421 s |
| First burn | ≈ 147 s → Earth parking orbit |
| Second burn (TLI) | ≈ 347 s, Δv ≈ 3,050 m/s |

**Command and Service Module (CSM-107 "Columbia")**
| Parameter | Value |
|---|---|
| Mass at TLI | ≈ 30,320 kg |
| Service Propulsion System (SPS) thrust | 91,190 N |
| SPS Isp | ≈ 314 s |
| Restartable | Yes — used for LOI-1, LOI-2, TEI and corrections |

## Apollo 11 mission timeline (Ground Elapsed Time)

Implement these as the reference profile. The user should be able to fly it and compare
their own performance against these marks.

| GET | Event |
|---|---|
| T+00:00:00 | Liftoff |
| T+00:00:12 | Tower clear, roll and pitch program begins |
| T+00:01:06 | Max Q, ≈ 13.8 km |
| T+00:02:15 | S-IC centre engine cutoff |
| T+00:02:41 | S-IC cutoff and staging |
| T+00:09:08 | S-II cutoff and staging |
| T+00:11:42 | S-IVB cutoff — parking orbit 190.8 × 183.2 km, 32.6° inclination |
| T+02:44:16 | Trans-Lunar Injection, S-IVB restart, ≈ 5 min 47 s burn |
| T+03:17 | Transposition, docking and LM extraction |
| T+75:50 | Lunar Orbit Insertion 1 — SPS, ≈ 357 s, Δv ≈ 889 m/s → 314 × 111 km |
| T+80:12 | LOI-2 — circularise to ≈ 122 × 100 km |
| T+100:12 | LM undocking |
| T+101:36 | Descent Orbit Insertion — Δv ≈ 23 m/s → 111 × 15.2 km |
| T+102:33 | Powered Descent Initiation — P63 begins |
| T+102:38 | First 1202 alarm, ≈ 3 km altitude |
| T+102:45:40 | Touchdown, Sea of Tranquility (0.674°N, 23.473°E) |
| T+124:22 | Lunar ascent — APS, ≈ 435 s burn |
| T+128:03 | Rendezvous and docking with CSM |
| T+135:23 | Trans-Earth Injection — SPS, ≈ 151 s, Δv ≈ 1,000 m/s |
| T+195:03 | Entry interface, 122 km, ≈ 11,000 m/s |
| T+195:18 | Splashdown, 24 July 1969 |

## Delta-v budget to verify against

The simulator should compute these itself and let the user compare:

| Leg | Δv |
|---|---|
| Earth surface → LEO | ≈ 9,400 m/s (of which ≈ 1,500–2,000 m/s is gravity and drag loss) |
| LEO → TLI | ≈ 3,120 m/s |
| TLI → Lunar Orbit Insertion | ≈ 890 m/s |
| Lunar orbit → surface (DOI + PDI) | ≈ 2,050 m/s |
| Lunar surface → lunar orbit | ≈ 1,870 m/s |
| Lunar orbit → Trans-Earth Injection | ≈ 1,000 m/s |

## Phase 8 — Launch and ascent to orbit

Implement a three-stage Saturn V ascent from the pad to a 185 km parking orbit.

Requirements:
- Full atmospheric flight: drag, dynamic pressure, Max Q identification and display.
- **Gravity turn** guidance: vertical rise, pitch-over manoeuvre, then a
  near-zero-angle-of-attack trajectory. Let the user tune the pitch program and see the
  cost of getting it wrong.
- Staging events with mass jettison; the S-IC dry mass must actually be discarded.
- Throttle-down at Max Q and centre-engine cutoff for the 4 g limit.
- Track and display gravity loss, drag loss, and steering loss separately from useful Δv.
- Orbit insertion check: is the resulting orbit stable, and what are its apogee and
  perigee?

*Learning goal: understand why reaching orbit costs 9,400 m/s when orbital velocity is
only 7,800 m/s.*

## Phase 9 — Trans-Lunar Injection and coast

- S-IVB restart, TLI burn, hyperbolic-relative escape from Earth's sphere of influence.
- Patched-conic transition from Earth-centred to Moon-centred reference frame.
- Three-day coast with time acceleration (1× up to 10,000×).
- Mid-course correction manoeuvres — small burns with large downstream effects. Make the
  user feel this sensitivity.
- Free-return trajectory option: the safety profile that brought Apollo 13 home.

## Phase 10 — Transposition, docking and extraction

- CSM separates, rotates 180°, docks nose-to-nose with the LM, and extracts it from the
  S-IVB adapter.
- Manual six-degree-of-freedom RCS control with docking alignment tolerances.
- This is a docking minigame with real relative-motion physics.

## Phase 11 — Lunar orbit operations

- LOI-1 and LOI-2 burns using the SPS.
- Undocking and Descent Orbit Insertion.
- Hand off to the existing P63–P67 descent code from the first simulator. **Reuse, do not
  rewrite.** If the handoff is awkward, that is a signal the earlier physics core was not
  cleanly separated — fix the architecture rather than duplicating code.

## Phase 12 — Ascent, rendezvous and docking

- Ascent Propulsion System: 15,600 N, Isp ≈ 311 s, ascent stage mass ≈ 4,700 kg.
- Staging: the descent stage stays on the surface as a launch platform.
- Insertion into a 17 × 87 km orbit, then coelliptic rendezvous with the CSM.
- Implement the **Clohessy-Wiltshire equations** for relative motion in orbit.
- Include the counter-intuitive result explicitly: thrusting forward raises your orbit and
  makes you *slower* in angular terms, so you fall behind. Make the user discover this.

## Phase 13 — Return and re-entry

- Trans-Earth Injection burn.
- Service Module jettison; only the Command Module returns.
- **Atmospheric entry at ≈ 11,000 m/s** — the hardest part of the mission.
- Entry corridor: too shallow and you skip off the atmosphere, too steep and you exceed
  the g-limit and burn through. The corridor is roughly 2° wide. Model this honestly.
- Peak heating and peak deceleration (Apollo peaked near 6.5 g).
- Lift-vector control by rolling the capsule — the actual technique used.
- Drogue and main parachute deployment, splashdown.

---

# PART B — THE MARS TWIST

Now the alternate-history campaign. The premise: the same engineering philosophy is
extended to Mars, using orbital propellant depots and in-situ resource production instead
of a single enormous vehicle.

This part is a **mission-design problem**, not a piloting problem. The core question the
simulator should let the user answer is: *what architecture actually closes?*

## Mars constants

| Parameter | Value |
|---|---|
| Surface gravity | 3.721 m/s² |
| Radius | 3,389,500 m |
| Gravitational parameter (μ) | 4.282837e13 m³/s² |
| Surface pressure | ≈ 610 Pa (0.6 % of Earth) |
| Atmospheric composition | ≈ 95 % CO₂ |
| Atmospheric scale height | ≈ 11,100 m |
| Sol length | 24 h 39 m 35 s |
| Semi-major axis | 1.524 AU |
| Orbital period | 687 Earth days |
| Earth-Mars synodic period | ≈ 780 days (25.6 months) |

## Mars mission delta-v budget

| Leg | Δv |
|---|---|
| LEO → Trans-Mars Injection | ≈ 3,600 m/s |
| Mars Orbit Insertion (propulsive) | ≈ 2,100 m/s |
| Mars Orbit Insertion (aerocapture) | ≈ 900 m/s saved |
| Mars orbit → surface (with EDL) | ≈ 700 m/s propulsive after atmospheric braking |
| Mars surface → Mars orbit | ≈ 4,100 m/s |
| Mars orbit → Trans-Earth Injection | ≈ 2,500 m/s |

Hohmann transfer time: ≈ 259 days each way. Because the return window does not open
immediately, a conjunction-class mission means roughly 500 days on the surface — total
mission duration near 900 days. Make the simulator show this brutally, because it is the
single fact that drives every architectural decision.

## Phase 14 — Mission architecture calculator

Before flying anything, build the tool that decides whether a mission is possible.

- Stage editor: define stages with dry mass, propellant mass, engine Isp and thrust.
- Compute total Δv via Tsiolkovsky and compare against the mission requirement.
- Payload-fraction analysis: show how mass fraction dominates everything.
- Demonstrate the tyranny of the rocket equation — plot required initial mass against
  required Δv and watch it go exponential.
- Let the user try a single-launch Mars mission and discover it does not close. **This
  failure is the point of the phase.**

## Phase 15 — Orbital propellant depots

This is the physically correct version of the "fuel station" idea.

**Depot locations to model** — and the simulator must explain why these and not arbitrary
points in space:
- **Low Earth Orbit depot** — cheapest to fill from Earth, but high boil-off exposure and
  significant station-keeping drag.
- **Earth-Moon L1** (≈ 326,000 km from Earth) and **Earth-Moon L2** (≈ 449,000 km) —
  gravitationally balanced, station-keeping cost only about 5–10 m/s per year. Ideal
  staging nodes.
- **Sun-Earth L1 and L2** (≈ 1.5 million km) — for deep-space staging.
- **Low Mars Orbit depot** — filled from Martian surface production.

**Boil-off — the problem that kills naive depot designs**
- Liquid hydrogen boils at 20 K and loses roughly 0.1–1 % per day without active cooling.
  A depot that sits for a two-year transfer window is empty when you arrive.
- Model this explicitly. Let the user watch a hydrogen depot drain.
- Then give them the engineering answers: zero-boil-off cryocoolers, sun shields, and
  switching to **methalox** — liquid methane boils at 111 K instead of 20 K, and Mars can
  manufacture it.
- Storable propellants (hydrazine/NTO) do not boil off at all but pay for it with much
  lower Isp. Make the trade visible.

**Depot mechanics to implement**
- Tanker launches from Earth, rendezvous and transfer.
- Transfer losses and residuals.
- Station-keeping budget over time.
- A depot network graph: nodes with propellant inventory, edges with Δv cost. Then let the
  user route a mission through it and see whether the architecture closes.

## Phase 16 — Interplanetary transfer

- Hohmann transfer calculation from Earth orbit to Mars orbit.
- **Porkchop plot generator**: for a grid of departure and arrival dates, compute required
  C3 and arrival velocity, and render it as a contour plot. This is the actual tool
  mission designers use, and building one teaches more orbital mechanics than any textbook
  chapter.
- Launch window identification against the 780-day synodic cycle.
- Lambert's problem solver for arbitrary transfer times.
- Gravity assists: model a Venus flyby option and compare it against the direct route.
- Bi-elliptic transfers, and the specific conditions under which they beat Hohmann.

## Phase 17 — Mars Entry, Descent and Landing

Completely different from a lunar landing, and the most interesting engineering problem in
the whole project. Mars has enough atmosphere to require a heat shield, but not enough to
land on parachutes alone.

Sequence to implement:
- Entry interface at ≈ 125 km, ≈ 5,500–7,500 m/s.
- Hypersonic phase: heat shield, peak heating, peak deceleration of 8–12 g.
- Guided entry using bank-angle modulation for downrange control.
- Supersonic parachute deployment around Mach 1.7–2.2, at ≈ 10 km.
- Heat shield jettison, radar altimeter acquisition.
- Powered descent: retropropulsion for the final phase, because parachutes cannot slow a
  heavy vehicle enough in that thin an atmosphere.
- Landing accuracy: model the ellipse and let the user try to shrink it.
- Communication delay of 4 to 24 minutes each way — **so this must be fully autonomous.
  No manual piloting is possible.** The user writes the guidance logic instead of flying
  it. That constraint is the lesson.

## Phase 18 — In-Situ Resource Utilisation

The real answer to "where does the return propellant come from".

- **Lunar ISRU**: water ice in permanently shadowed craters at the south pole (Shackleton
  and similar). Electrolysis of water into hydrogen and oxygen. Model the power budget —
  roughly 50–55 kWh per kg of hydrogen produced in practice.
- **Mars ISRU**: the atmosphere is 95 % CO₂. Model the **Sabatier reaction**
  (CO₂ + 4H₂ → CH₄ + 2H₂O, exothermic) to produce methane, plus solid-oxide electrolysis
  for oxygen. NASA's MOXIE experiment on Perseverance demonstrated the oxygen half of this
  in flight.
- Power systems: solar array area versus dust accumulation and seasonal dust storms,
  against a nuclear surface reactor.
- Production rate versus mission timeline: can you fill the return vehicle before the
  window closes? Make this a scheduling constraint the user has to solve.

## Phase 19 — Campaign mode

Tie it together. The user plans and executes a multi-launch Mars campaign:
- Pre-position depots and cargo during earlier windows.
- Manage propellant inventory and boil-off across the whole network over years.
- Schedule crew departure against the synodic cycle.
- Handle failures: a tanker launch fails, a depot leaks, a dust storm cuts ISRU output.
- Win condition: crew launched, landed, resupplied and returned to Earth.

---

# ADVANCED PROPULSION — REAL OPTIONS ONLY

If the user unlocks better engines, these are the ones that exist. Model each with honest
numbers.

| Technology | Isp | Thrust | Honest assessment |
|---|---|---|---|
| Chemical (LH2/LOX) | 450 s | High | Baseline. Boil-off problem. |
| Chemical (methalox) | 380 s | High | Lower Isp, but storable and Mars can make it. Usually wins. |
| **Nuclear thermal (NERVA-class)** | 825–900 s | ≈ 334 kN | Roughly doubles chemical Isp. A reactor heats hydrogen — it still expels reaction mass, the reactor only supplies heat. Currently being revived under the DRACO programme. |
| **Ion / gridded electrostatic** | 3,000–5,000 s | Milli-newtons | Extremely efficient, extremely weak. Months of continuous thrust. This is the one case where electromagnetic fields genuinely produce thrust — by accelerating ionised propellant, not by pushing against a planetary field. |
| **Hall-effect thruster** | 1,500–2,500 s | 10s–100s of mN | Practical workhorse for station-keeping and slow cargo transfer. |
| **Solar sail** | No propellant | ≈ 9 μN/m² at 1 AU | Real — IKAROS flew it. Thrust falls with the inverse square of solar distance, so it weakens badly beyond Mars. Model a large sail and let the user see how long a transfer actually takes. |
| **Magnetic sail / solar wind** | No propellant | Minute | Include only if modelled honestly. The solar wind carries far less momentum flux than sunlight does. Do not overstate it. |

---

# WHAT I WANT FROM YOU

1. **Ask me which single phase I am building before writing any code.** If I paste this
   whole document, tell me to pick one phase instead.
2. Deliver complete, runnable code for that phase only — no stubs in the physics.
3. Reuse the existing physics core, vehicle model and DSKY code. If reuse is difficult,
   tell me the architecture needs refactoring first and propose the refactor.
4. Explain the physics and the mission-design reasoning behind every equation, in comments
   and in your reply.
5. Correct me directly whenever I state something physically wrong, and explain why.
6. After each phase, give me a manual verification checklist and one hand-calculation I
   can do with a calculator to confirm the numbers are right.
7. Tell me honestly if a phase is too large and should be split further.

Start by asking which phase I want to build.

## PROMPT ENDS HERE

---

## Notes for you, Kanak — not part of the prompt

**On sequencing.** You are on Phase 1. The gap between a working vertical lander and a
Mars campaign simulator is roughly two years of consistent evening work. That is not a
discouragement — it is the honest scale, and it is exactly the kind of long project that
makes an MCA thesis and a portfolio centrepiece. But the way to get there is Phase 2 next
week, not Phase 14 tonight.

**The phase that matters most for your career.** Phase 16, the porkchop plot generator.
It is pure orbital mechanics plus numerical methods plus data visualisation — all three
things you want on your CV, in one self-contained tool. It also stands alone, so you could
build it independently if you ever want a break from the lander.

**The phase that will teach you the most.** Phase 14, the architecture calculator. When
you build the tool that proves a single-launch Mars mission cannot close, the rocket
equation stops being an equation and becomes an intuition. Everything about depots and
ISRU follows from that one result.

**On the Mars "twist".** Your original instinct — fuel stations in space — was correct.
It is a genuinely serious proposal that space agencies and companies study. What changed
is only the *where* and the *why*: Lagrange points rather than arbitrary orbits, because
station-keeping there costs almost nothing. Good instinct, corrected placement. That is
how engineering usually goes.

**Architecture advice you should act on now.** Phase 11 requires handing off from
interplanetary flight to your existing P63 descent code. If your Phase 1–7 physics is
cleanly separated from pygame, this will be easy. If it is tangled into the render loop,
it will be painful. Keep `physics.py` pure — state in, derivative out, no drawing, no
input, no global state. Every hour you spend keeping that boundary clean now saves a day
later.
