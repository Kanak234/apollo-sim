# MASTER BUILD PROMPT — Apollo 11 Full Mission Simulator, Extended to Mars

> **One document, fifteen phases.** Paste this entire file once at the start of a project.
> The execution protocol in Section 0 tells the assistant to work through it phase by phase
> and stop for your confirmation between each one. You do not need to paste anything again —
> you just say "Phase N" when you are ready.
>
> Phase 1 is already complete. Start from Phase 2.

---

## PROMPT BEGINS HERE

You are an experienced flight-software and simulation engineer with a background in
astrodynamics. I am a BCA student with strong Python skills and working ML/CV pipelines
behind me, but limited formal physics training. Explain the physics as you implement it —
I want to understand every equation, not just run the code.

I am building a **complete Apollo 11 mission simulator** in Python — Earth launch to
splashdown, historically accurate — and then extending the same simulation engine to a
**Mars mission architecture with orbital propellant depots**.

This is a serious engineering learning project, not a game. The physics must be correct,
the constants must be real and sourced, and the interface should echo the actual Apollo
Guidance Computer.

---

### 0. EXECUTION PROTOCOL — READ THIS FIRST

This document specifies fifteen phases. **Do not attempt to build more than one phase per
response.** Follow this protocol strictly:

1. On receiving this document, acknowledge it, summarise the phase list in two lines, and
   ask which phase I want to build. Do not write code yet.
2. When I name a phase, build **that phase only** — complete, runnable files with no
   placeholders and no `TODO` stubs in the physics.
3. After delivering a phase: give me a manual verification checklist, and one specific
   quantity I should calculate by hand or plot to confirm the physics is right.
4. Then **stop and wait.** Do not begin the next phase until I say so.
5. If a later phase requires refactoring earlier code, say so explicitly and explain what
   changes and why, before making the change.
6. Where I state an incorrect physical assumption anywhere in this document or in
   conversation, correct me directly and explain the correct physics.

Phases 1–7 build the lunar descent. Phases 8–12 build the full historical mission around
it. Phases 13–15 are the Mars extension. **Phase 1 is already complete.**

---

### 1. NON-NEGOTIABLE PHYSICS REQUIREMENTS

These apply to every phase.

**Newtonian dynamics**
- F = ma with **variable mass** throughout. Mass is never constant once propellant flows.
- Vectors in the simulation plane. From Phase 8, use a planet-centred inertial frame.
- Atmospheric drag where an atmosphere exists (Earth launch, Earth re-entry, Mars EDL);
  none on the Moon or in vacuum.

**Variable mass and the rocket equation**
- `dm/dt = -F_thrust / (Isp * g0)` with `g0 = 9.80665 m/s²`.
  `g0` is Earth's standard gravity and appears in the definition of specific impulse
  regardless of where the vehicle is — this is a common confusion, comment it clearly.
- Tsiolkovsky: `Δv = Isp * g0 * ln(m_initial / m_final)`.
- Maintain a live delta-v budget: capability remaining versus mission requirement. When
  the two cross, the mission is unrecoverable. Show this to the user.

**Gravity**
- Inverse-square from Phase 4 onward: `a = -μ * r_vec / |r|³`.
- Constant-gravity approximation permitted only in Phases 1–3.
- From Phase 9, model gravity from more than one body (patched conics is acceptable;
  full n-body is a stretch goal).

**Numerical integration**
- Hand-written **RK4** with fixed timestep. No imported ODE solvers — writing the
  integrator is a learning objective.
- Physics timestep decoupled from render framerate; simulation must be deterministic.
- Adaptive timestep from Phase 9 onward: long coasts need large steps, burns and
  atmospheric flight need small ones. Explain the stability considerations.

**Explicitly do NOT implement**
- Magnetic or "anti-gravity" propulsion. Gravity and magnetism are unrelated forces and a
  spacecraft is not ferromagnetic — a magnetic field cannot push it away from a planet.
  All thrust comes from expelling reaction mass, per Newton's third law.
- Infinite or arcade-style fuel. Propellant is finite and mass-coupled everywhere.
- Any propellant depot that ignores boil-off. See Phase 13.

---

### 2. HISTORICAL ACCURACY MANDATE (PHASES 8–12)

Phases 8 through 12 reconstruct Apollo 11 **as it actually flew**. Use the real timeline,
the real burn durations, and the real vehicle parameters below. Where a value is
approximate or disputed, say so in a comment rather than inventing precision.

Mission clock is **MET** (Mission Elapsed Time) from liftoff, displayed as `hhh:mm:ss`.

| Event | MET | Detail |
|---|---|---|
| Liftoff, LC-39A | 000:00:00 | 16 July 1969, 13:32:00 UTC |
| S-IC cutoff / staging | ≈ 000:02:42 | ~67 km altitude, ~2,760 m/s |
| S-II cutoff / staging | ≈ 000:09:12 | ~185 km |
| S-IVB cutoff, parking orbit | ≈ 000:11:40 | 185.6 × 187.4 km, 32.5° inclination |
| Translunar Injection (TLI) | ≈ 002:44:16 | S-IVB restart, ~347 s burn, Δv ≈ 3,050 m/s |
| Transposition, docking, extraction | ≈ 003:17:00 | CSM turns, docks with LM, pulls it clear |
| Lunar Orbit Insertion (LOI-1) | ≈ 075:49:50 | ~357 s burn, Δv ≈ 889 m/s → 314 × 111 km |
| LOI-2 circularisation | ≈ 080:11:36 | ~17 s burn → ~110 km circular |
| Undocking | ≈ 100:12:00 | Eagle separates from Columbia |
| Descent Orbit Insertion (DOI) | ≈ 101:36:14 | ~30 s burn, Δv ≈ 23 m/s → 15.2 km perilune |
| Powered Descent Initiation (PDI) | ≈ 102:33:05 | Start of the 12.5-minute descent |
| 1202 alarm | ≈ 102:38:26 | First executive overflow, ~5 min into descent |
| Touchdown | 102:45:40 | 20:17:40 UTC, ~25 s propellant margin |
| First EVA step | ≈ 109:24:00 | |
| Lunar liftoff (APS ignition) | ≈ 124:22:00 | ~435 s burn, Δv ≈ 1,850 m/s |
| Rendezvous and docking | ≈ 128:03:00 | |
| Trans-Earth Injection (TEI) | ≈ 135:23:42 | ~151 s SPS burn, Δv ≈ 1,000 m/s |
| Entry interface | ≈ 195:03:06 | 121.9 km altitude, ~11,000 m/s |
| Splashdown | 195:18:35 | 24 July 1969 |

---

### 3. VEHICLE CONSTANTS

Put all of these in `constants.py` with units and source comments.

**Saturn V — S-IC first stage**
| Parameter | Value |
|---|---|
| Engines | 5 × F-1, RP-1 / LOX |
| Thrust (sea level, total) | ≈ 33,400 kN |
| Isp | 263 s sea level, 304 s vacuum |
| Propellant mass | ≈ 2,077,000 kg |
| Burn duration | ≈ 168 s |
| Dry mass | ≈ 131,000 kg |

**Saturn V — S-II second stage**
| Parameter | Value |
|---|---|
| Engines | 5 × J-2, LH2 / LOX |
| Thrust (vacuum, total) | ≈ 5,000 kN |
| Isp | ≈ 421 s vacuum |
| Propellant mass | ≈ 456,000 kg |
| Burn duration | ≈ 384 s |
| Dry mass | ≈ 36,000 kg |

**Saturn V — S-IVB third stage**
| Parameter | Value |
|---|---|
| Engines | 1 × J-2, restartable |
| Thrust (vacuum) | ≈ 1,000 kN |
| Isp | ≈ 421 s vacuum |
| Propellant mass | ≈ 108,000 kg |
| Burn 1 (to parking orbit) | ≈ 147 s |
| Burn 2 (TLI) | ≈ 347 s |
| Dry mass | ≈ 13,500 kg |

**Full stack at liftoff:** ≈ 2,970,000 kg, height 110.6 m.

**Command and Service Module (CSM-107 "Columbia")**
| Parameter | Value |
|---|---|
| Total mass at TLI | ≈ 30,320 kg |
| Service Propulsion System (SPS) thrust | 91,190 N |
| SPS Isp | ≈ 314 s |
| SPS propellant | ≈ 18,410 kg |
| Command Module mass at entry | ≈ 5,560 kg |

**Lunar Module (LM-5 "Eagle")**
| Parameter | Value |
|---|---|
| Total mass at PDI | ≈ 15,100 kg |
| Descent stage usable propellant | ≈ 8,200 kg |
| DPS max thrust | 45,040 N |
| DPS throttle range | 10 % – 60 %, plus fixed full |
| DPS Isp | ≈ 311 s |
| Ascent stage mass at liftoff | ≈ 4,700 kg |
| APS thrust | 15,600 N |
| APS Isp | ≈ 311 s |
| RCS thruster force | ≈ 445 N per quad |

**Landing gear limits — exceed any and it is a crash:**
vertical ≤ 3.0 m/s · horizontal ≤ 1.2 m/s · tilt ≤ 12° · propellant > 0 kg.
Apollo 11 touched down at ≈ 0.5 m/s. That is the standard.

**Bodies**
| Body | μ (m³/s²) | Radius (m) | Surface gravity |
|---|---|---|---|
| Earth | 3.986004418e14 | 6,371,000 | 9.80665 m/s² |
| Moon | 4.9028695e12 | 1,737,400 | 1.625 m/s² |
| Mars | 4.282837e13 | 3,389,500 | 3.721 m/s² |
| Sun | 1.32712440018e20 | 695,700,000 | — |

Earth–Moon distance ≈ 384,400 km. Earth sidereal rotation 86,164.1 s.

---

### 4. THE DSKY INTERFACE

Reproduce the look and logic of the Display and Keyboard unit — not the actual AGC source.

**Layout:** `PROG` / `VERB` / `NOUN` indicator fields (2 digits each); registers `R1`,
`R2`, `R3` (5 signed digits each); warning lamps `PROG`, `RESTART`, `GIMBAL LOCK`,
`NO ATT`, `VEL`, `ALT`, `UPLINK ACTY`. Green-on-black electroluminescent styling,
fixed-width font.

**Verb/Noun set**

| Command | Displays |
|---|---|
| V06 N62 | R1 velocity magnitude, R2 altitude rate, R3 altitude |
| V16 N68 | Landing monitor: R1 slant range, R2 time-to-go, R3 altitude rate |
| V16 N63 | R1 total velocity, R2 altitude rate, R3 altitude |
| V06 N60 | R1 forward velocity, R2 altitude rate, R3 altitude |
| V05 N09 | Active alarm codes |
| V37 Enn | Change program (V37 E63 selects P63) |
| V16 N65 | Mission elapsed time across R1/R2/R3 |
| V06 N94 | Propellant remaining, estimated burn time left |
| V82 | Request orbital parameters (apoapsis / periapsis / period) |
| V16 N44 | Orbit monitor: R1 apoapsis, R2 periapsis, R3 time to node |

Keys: `VERB`, `NOUN`, `PROG`, `0`–`9`, `+`, `-`, `ENTR`, `CLR`, `KEY REL`, `RSET`.
The `V16` prefix means *monitor* — refresh continuously until another command is entered.

**Programs**

| Program | Name | Behaviour |
|---|---|---|
| P11 | Earth Orbit Insertion Monitor | Ascent monitoring during Saturn V flight |
| P15 | TLI | Translunar injection targeting |
| P30 | External Δv | Manual burn parameter entry |
| P40 | SPS Burn | Service Propulsion System burn execution |
| P63 | Braking Phase | High-thrust retrograde burn killing orbital velocity |
| P64 | Approach Phase | Pitch-over; Landing Point Designator becomes usable |
| P65 | Automatic Landing | Guidance flies terminal descent |
| P66 | Rate of Descent (ROD) | **Manual — what Armstrong actually used.** Each input shifts target descent rate by ±0.3 m/s; computer holds it |
| P67 | Manual Full | Direct throttle and attitude, no guidance |
| P12 | Powered Ascent | Lunar liftoff to orbit |
| P32–P35 | Rendezvous sequence | CSI, CDH, TPI, TPM targeting |
| P61–P67E | Entry programs | Re-entry guidance and monitoring |

**The 1202 alarm — implement this properly, it is the most instructive part.**
Model a computation budget: each active task consumes cycles per tick. When demand exceeds
capacity, raise **1202 EXECUTIVE OVERFLOW** (or **1201 NO CORE SETS**), light the `PROG`
lamp, and trigger a software restart that reloads state and sheds the lowest-priority
tasks — exactly as the real AGC's restart protection did. Critical guidance must survive.
Print an explanation to the console: the alarm was not a failure, it was the computer
correctly shedding load and recovering.

---

### 5. MODULE ARCHITECTURE

Grow this structure as phases are added. Do not put everything in one file.

```
apollo_sim/
├── constants.py          # All constants, units, sources
├── bodies.py             # Celestial bodies, ephemerides, reference frames
├── physics.py            # State vector, derivatives, RK4, adaptive stepping
├── atmosphere.py         # Earth and Mars atmosphere models, drag
├── vehicle.py            # Base vehicle: mass, propellant, thrust, attitude
├── stages.py             # Multi-stage vehicle, staging events, mass fractions
├── guidance.py           # All P-programs, gravity turn, PEG, ROD controller
├── navigation.py         # State estimation, Kalman filter, sensor models
├── agc.py                # Executive scheduler, priorities, 1202/1201, restart
├── dsky.py               # DSKY rendering and verb/noun parsing
├── orbit.py              # Orbital elements, propagation, Kepler solver, transfers
├── rendezvous.py         # Clohessy-Wiltshire, rendezvous targeting
├── entry.py              # Atmospheric entry, corridor, heating, parachutes
├── depot.py              # Propellant depots, boil-off, transfer operations (Phase 13+)
├── mission.py            # Mission sequencer, event timeline, MET clock
├── display.py            # Rendering: world view, HUD, DSKY, instruments
├── telemetry.py          # CSV logging of full state each timestep
├── analysis.py           # Post-flight plots and comparison to real Apollo data
├── main.py               # Entry point
└── tests/                # pytest suite — one file per physics module
```

**Code requirements**
- Type hints throughout; docstrings on every public function.
- Physics functions **pure**: state in, derivative out, no side effects. Testable.
- `pytest` coverage for every phase: analytical comparisons, conservation laws,
  hand-calculated checks.
- Comment the **physics**, not the syntax. Explain why an equation has the form it has.

---

### 6. PHASES 1–7 — LUNAR DESCENT CORE

**Phase 1 — Minimum viable lander. ✅ COMPLETE.**
1D vertical, constant gravity, constant mass, on/off thrust, crash-or-land verdict.

**Phase 2 — Real propulsion**
Finite propellant; mass depletion coupled to thrust; throttleable DPS in the real 10–60 %
band; Isp-based flow rate; delta-v remaining display; fuel-exhaustion failure mode.
*Goal: understand why mass fraction dominates every rocket design decision.*
*Note: with constant mass and on/off thrust, hovering was impossible. Throttle changes that
— observe the difference.*

**Phase 3 — Two dimensions and attitude**
Horizontal velocity; vehicle rotation; thrust vectored along the body axis; RCS
translation; all three touchdown limits enforced; terrain with a designated landing site.
*Goal: realistic control difficulty.*

**Phase 4 — Orbital mechanics**
Start from a real 15.2 km perilune at ≈ 1,690 m/s. Inverse-square gravity. Implement P63
braking, P64 approach, P66 manual ROD, with the correct descent timeline.
*Goal: a genuine Apollo descent profile, not a hover puzzle.*

**Phase 5 — The AGC layer**
Full DSKY with verb/noun entry, program switching, warning lamps, executive task
scheduler, 1202/1201 alarms and restart logic.
*Goal: understand how flight software degrades gracefully instead of failing.*

**Phase 6 — Autopilot and state estimation**
PID controller for automatic descent-rate hold (P65). **Kalman filter** estimating altitude
and velocity from deliberately noisy simulated radar-altimeter and IMU measurements. Let
the user toggle raw versus filtered state to see the difference.
*Goal: state estimation is the core of guidance, navigation and control — the most
employable skill in this entire project.*

**Phase 7 — Analysis tooling**
CSV telemetry logging; post-flight plots (altitude, velocity, mass, thrust, ground track);
comparison against the real Apollo 11 descent timeline.

---

### 7. PHASES 8–12 — THE COMPLETE HISTORICAL MISSION

**Phase 8 — Earth launch and ascent to parking orbit**

Simulate the Saturn V from liftoff to a 185 km circular parking orbit.

- Three-stage vehicle with real masses, thrusts, Isp values and staging events.
- **Gravity turn:** vertical rise for the first few seconds, then a pitch programme that
  lets gravity rotate the velocity vector. Explain why an immediate horizontal turn would
  be structurally destructive and why a purely vertical ascent wastes enormous delta-v.
- **Atmospheric drag** using an exponential atmosphere model (scale height ≈ 8,500 m).
  Model **max Q**, the point of maximum dynamic pressure, at roughly 13–14 km — and the
  throttle-down that Apollo actually performed through it.
- **Gravity losses** and **drag losses** tracked separately and displayed. Earth surface to
  LEO needs ≈ 9,400 m/s of delta-v while orbital velocity is only ≈ 7,800 m/s — the
  difference is these losses. Make the user see that number grow in real time.
- Staging: dropped mass, ullage, engine restart.
- Earth rotation contributes ≈ 408 m/s eastward at Cape Canaveral's latitude — include it
  and explain why launch sites cluster near the equator and fire east.
- Orbital insertion accuracy check: report resulting apoapsis, periapsis, inclination.

*Verify by hand: the ideal delta-v from the rocket equation across all three stages,
compared against 9,400 m/s. The gap is your losses.*

**Phase 9 — Translunar injection and coast**

- S-IVB restart for TLI: ≈ 3,050 m/s, raising apogee to lunar distance.
- **Transposition, docking and extraction:** CSM separates, rotates 180°, docks with the
  LM, and pulls it free of the S-IVB. Implement as a docking minigame with real RCS
  authority.
- Translunar coast with **patched-conic** approximation: Earth's sphere of influence
  (radius ≈ 924,000 km), then the Moon's (radius ≈ 66,100 km).
- Free-return trajectory concept: explain why the early Apollo flights used it and what it
  bought them.
- Mid-course corrections: small burns, large downstream effect. Let the user compute and
  execute one.
- Adaptive timestep becomes essential here — a three-day coast at 0.01 s steps is
  unusable. Explain the accuracy trade-off.

**Phase 10 — Lunar orbit insertion and descent handoff**

- LOI-1: SPS retrograde burn on the far side, out of contact with Earth. Explain why the
  burn had to happen there and what failure would have meant.
- LOI-2 circularisation.
- Undocking and DOI.
- Hand off cleanly to the Phase 4 descent code. This is the integration test for the whole
  architecture — if the descent module was written purely, it should drop in unchanged.

**Phase 11 — Ascent, rendezvous and docking**

- APS ignition from the descent stage used as a launch platform. Ascent to ≈ 18 km
  perilune orbit, then circularisation.
- **Rendezvous** using the **Clohessy-Wiltshire equations** for relative motion in orbit.
  Implement the CSI, CDH and TPI burn sequence.
- Teach the counter-intuitive result explicitly: to catch a target ahead of you, you must
  **slow down** to drop into a lower, faster orbit. Make the user experience this.
- Final approach and docking with RCS translation only.

**Phase 12 — Trans-Earth injection, re-entry and splashdown**

- TEI burn: ≈ 1,000 m/s, escaping lunar orbit.
- Trans-Earth coast with mid-course corrections.
- CM/SM separation.
- **Atmospheric entry** at ≈ 11,000 m/s. Entry interface at 121.9 km.
- **The entry corridor:** flight path angle of −6.5° ± 1°. Too shallow and the capsule
  skips off the atmosphere into a long orbit; too steep and deceleration exceeds survivable
  limits. Model both failure modes and let the user hit them.
- Lifting entry: the CM flew offset centre-of-mass to generate a small lift vector, rolled
  to steer. Implement roll control for downrange targeting.
- Deceleration profile peaking around 6.5 g; heating rate proportional to roughly
  `ρ^0.5 · v^3`. Display a heat-shield margin.
- Drogue and main parachute deployment at the correct altitudes, then splashdown accuracy
  scoring.

---

### 8. PHASES 13–15 — THE MARS EXTENSION

Phases 8–12 reconstruct history. **Phases 13–15 are a counterfactual architecture study.**
Keep that distinction visible in the code and the UI: this is not what happened, it is what
the physics permits. Label these phases clearly as speculative-but-physical.

**Phase 13 — Orbital propellant depots**

This is the "fuel station in space" idea, done correctly.

*Depot locations — and why these and not others:*
- **LEO depot** (≈ 400 km): the cheapest place to deliver propellant from Earth, so it is
  where mass accumulates. Costs continuous drag makeup.
- **Earth–Moon L1**: gravitationally balanced staging node. Station-keeping in a halo orbit
  costs only a few m/s per year. From EML1, delta-v to the lunar surface ≈ 2,520 m/s, and
  departure toward Mars is cheap. **This is the physically correct answer to "where should
  the fuel station go" — not an arbitrary orbit.**
- **Sun–Earth L2**: deep-space staging, useful for departure and for cryogenic storage in
  permanent shade.
- **Mars orbit / Phobos or Deimos**: return-leg propellant, ideally produced locally.

*The central engineering problem — boil-off. Model it or the whole architecture is fiction:*
- Liquid hydrogen boils off at roughly **0.1 % – 1 % of stored mass per day** with passive
  insulation alone. Over a 260-day Mars transfer this can consume the entire propellant load.
- Model insulation performance, sun shields, and active cryocoolers with their electrical
  power cost.
- Give the user a real trade: store hydrogen (high Isp ≈ 450 s, severe boil-off) versus
  methane (lower Isp ≈ 370 s, far more storable, and manufacturable on Mars). Make them
  choose and live with the consequences.

*Depot operations to implement:*
- Rendezvous and berthing with the depot (reuse the Phase 11 rendezvous code).
- Propellant transfer in microgravity: settling thrust or surface-tension acquisition
  devices, transfer rate limits, ullage management.
- Depot inventory tracking, resupply scheduling, and a mass-balance ledger.
- Failure modes: leak, cryocooler failure, missed resupply window.

*Verify by hand: compute total delta-v for a direct Earth-to-Mars mission versus a
depot-staged one. Show the user the mass saving in the initial mass in low Earth orbit
(IMLEO) — this is the entire argument for depots, and it should appear as a number.*

**Phase 14 — Earth to Mars transfer**

- **Hohmann transfer:** ≈ 259 days Earth to Mars. Trans-Mars Injection from LEO
  ≈ 3,600 m/s.
- **Launch windows:** the Earth–Mars synodic period is ≈ 780 days, so a window opens
  roughly every 26 months. Implement a window calculator and make the user wait for one.
- **Porkchop plots:** generate departure-date versus arrival-date contours of required
  delta-v. This is the single most useful mission-design visualisation in existence and it
  is entirely within your reach to produce with matplotlib.
- **Lambert's problem:** solve for the transfer orbit connecting two positions in a given
  time. Implement a Lambert solver — this is the mathematical heart of interplanetary
  mission design.
- Patched conics: Earth SOI → heliocentric transfer → Mars SOI.
- Cruise-phase concerns: communication delay growing to 22 minutes one-way, solar
  conjunction blackout, radiation exposure accumulation, consumables.
- **Mars arrival, two options — make the user choose:**
  - *Propulsive capture:* ≈ 2,100 m/s of delta-v. Simple, expensive in propellant.
  - *Aerocapture:* use the atmosphere to shed velocity. Nearly free in propellant, but the
    corridor is narrow and the failure modes are unforgiving. Model both.

**Phase 15 — Mars entry, descent, landing, ISRU and return**

- **Mars EDL — the "seven minutes of terror".** Entry velocity ≈ 5,500–7,500 m/s.
- The core difficulty, stated plainly: the Martian atmosphere is roughly **0.6 %** of
  Earth's surface pressure (≈ 600 Pa) — thick enough to require a heat shield, far too thin
  to slow a heavy vehicle by parachute alone. Model the full sequence: heat shield →
  supersonic parachute → powered terminal descent. Explain why this combination is
  mandatory and why Mars landing masses have stayed small.
- Surface gravity 3.721 m/s²; landing gear limits comparable to the LM's.
- **ISRU — In-Situ Resource Utilisation**, the physically correct version of refuelling:
  - *Sabatier reaction:* `CO₂ + 4H₂ → CH₄ + 2H₂O`. Methane and water from Martian
    atmospheric carbon dioxide plus imported or mined hydrogen.
  - *Water ice electrolysis:* `2H₂O → 2H₂ + O₂`.
  - *Atmospheric oxygen production* — NASA's MOXIE experiment on Perseverance demonstrated
    this on the actual Martian surface.
  - Model power requirements, production rate, and time-to-fill. Make the return propellant
    a function of how long the crew stays and how much power they have.
- **Return leg:** ascent from Mars (escape velocity 5,027 m/s), Trans-Earth Injection,
  cruise, Earth entry at ≈ 12,000 m/s — faster and hotter than lunar return, with a
  correspondingly tighter corridor.
- **Full mission ledger:** total delta-v, total IMLEO, mission duration, propellant
  produced versus imported, crew radiation dose. Present it as a mission report the user
  can compare across different architecture choices.

---

### 9. TECHNOLOGY STACK

- **Python 3.11+**
- **pygame** for rendering and input — keep it isolated in `display.py` so the physics core
  stays reusable and headless-testable
- **NumPy** for vector and matrix mathematics
- **SciPy** permitted from Phase 14 for root-finding in the Lambert solver only — never for
  the trajectory integration itself
- **matplotlib** for analysis plots and porkchop plots
- **pytest** for the test suite

No game engine. No external physics or astrodynamics library. The physics is the project.

---

### 10. WHAT I EXPECT FROM YOU IN EVERY RESPONSE

1. Build one phase only, then stop.
2. Complete runnable files — no placeholders, no stubbed physics.
3. Explain the physics behind each equation, in comments and in your reply.
4. Correct my physical misconceptions directly when you encounter them.
5. Give me a manual verification checklist after each phase.
6. Give me one quantity to hand-calculate or plot that would prove the physics is right.
7. Flag honestly when a phase requires refactoring earlier code, before doing it.

Begin by acknowledging this document, summarising the phase list in two lines, and asking
which phase I want to build. Phase 1 is already complete.

## PROMPT ENDS HERE

---

## Notes for you, Kanak (not part of the prompt)

**On the "one prompt" question.** This is one document, but it is not one build. No
assistant can produce fifteen phases in a single response — context limits alone prevent
it, and anything that tried would give you fifteen broken half-modules instead of one
working simulator. What this file does instead is give the assistant the complete
architecture up front, so every phase is written knowing what comes later, while still
building them one at a time. That is what makes the module structure hold together.

**Phase order matters more now, not less.** Phase 8 (Earth launch) is genuinely harder than
everything in Phases 1–7 combined — the gravity turn, atmospheric drag and staging all
interact. Do not jump to it. Phase 2 next.

**The Mars phases are honest but speculative.** Phases 8–12 are history and you should
treat every number in them as checkable fact. Phases 13–15 are an architecture study. Keep
that line visible in your code and in anything you present — it is the difference between
an engineering project and science fiction, and reviewers notice.

**On boil-off.** If you build only one thing from Phase 13, build the boil-off model. Every
casual proposal for orbital fuel depots ignores it, and it is the reason depots remain
unbuilt rather than routine. Modelling it correctly is what will make this project look
like it was made by someone who read the literature.

**Portfolio framing.** When this is done, the strongest way to present it is not "I built a
lunar lander game." It is: "I implemented a full Apollo 11 mission profile from real
constants, validated the descent against the historical timeline, then used the same engine
to evaluate a depot-staged Mars architecture against a direct one." That sentence gets you
interviews.
