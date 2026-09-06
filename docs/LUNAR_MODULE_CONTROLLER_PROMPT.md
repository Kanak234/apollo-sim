# Build Prompt — Apollo-Style Lunar Module Controller Simulator

> **How to use this file.** Everything below the line is a single, self-contained prompt.
> Paste it into Claude, ChatGPT, Cursor, or any coding assistant. Do **not** paste all
> phases at once — paste the Core Spec plus **one phase**, get that working, then come
> back for the next phase. Building it in order is the whole point.

---

## PROMPT BEGINS HERE

You are an experienced flight-software and simulation engineer. I am a BCA student with
strong Python skills, working ML/CV pipelines behind me, but limited formal physics
training. Explain the physics as you implement it — I want to understand every equation,
not just run the code.

I am building an **Apollo-inspired Lunar Module descent simulator** in Python. This is a
serious engineering learning project, not a game. The physics must be correct, the
constants must be real, and the interface should echo the actual Apollo Guidance Computer.

---

### 1. PROJECT GOAL

Build a keyboard-controlled simulator in which the user manually flies an Apollo Lunar
Module from lunar orbit down to a soft landing on the Moon — using the same phased
descent structure (P63 → P64 → P66) that Apollo 11 used, and an on-screen DSKY-style
interface for reading spacecraft state.

Success is defined as: touching down within the landing-gear limits, with fuel remaining,
having flown the descent yourself.

---

### 2. NON-NEGOTIABLE PHYSICS REQUIREMENTS

Implement these correctly. Do not simplify them away.

**Newtonian dynamics**
- Motion governed by F = ma with a **variable-mass** vehicle.
- Thrust and gravity resolved as vectors in a 2D plane (downrange × altitude).
- No air resistance — the Moon has no meaningful atmosphere.

**Variable mass and the rocket equation**
- Vehicle mass decreases as propellant burns: `dm/dt = -F_thrust / (Isp * g0)`
  where `g0 = 9.80665 m/s²` (this is Earth's standard gravity — it appears in the Isp
  definition regardless of where the vehicle is, which is a common point of confusion;
  explain this in a comment).
- Thrust-to-weight ratio therefore rises during the burn. The vehicle gets more
  responsive as it empties.
- Track total delta-v capability remaining via the Tsiolkovsky equation:
  `Δv = Isp * g0 * ln(m_initial / m_final)`.

**Gravity**
- Lunar surface gravity: **1.625 m/s²**.
- Phase 1 may treat gravity as constant. From Phase 4 onward, use the inverse-square law
  with `μ_moon = 4.9028695e12 m³/s²` and `R_moon = 1737400 m`.

**Numerical integration**
- Use **fourth-order Runge-Kutta (RK4)** with a fixed timestep, not naive Euler.
- Implement RK4 by hand in a clearly commented function. Do not import an ODE solver —
  writing this myself is a learning objective.
- Decouple the physics timestep from the render framerate so the simulation is
  deterministic regardless of display performance.

**Explicitly do NOT implement**
- Any form of magnetic or "anti-gravity" propulsion. Gravity and magnetism are unrelated
  forces, and a spacecraft is not ferromagnetic — a magnetic field cannot push it away
  from a planet. All thrust in this simulator comes from expelling reaction mass, per
  Newton's third law.
- Arcade-style "hover forever" fuel behaviour. Propellant is finite and mass-coupled.

---

### 3. REAL VEHICLE CONSTANTS (Apollo 11, LM-5 "Eagle")

Define these in a dedicated `constants.py` with units and source comments.

| Parameter | Value |
|---|---|
| Lunar surface gravity | 1.625 m/s² |
| Lunar radius | 1,737,400 m |
| Lunar gravitational parameter (μ) | 4.9028695e12 m³/s² |
| LM mass at powered descent initiation | ≈ 15,100 kg |
| Descent stage usable propellant | ≈ 8,200 kg |
| Descent Propulsion System (DPS) max thrust | 45,040 N |
| DPS throttle range | 10 % – 60 %, plus a fixed full-thrust setting |
| DPS specific impulse (Isp) | ≈ 311 s |
| Reaction Control System (RCS) thruster force | ≈ 445 N per quad |
| Descent orbit perilune (start altitude) | ≈ 15,240 m |
| Orbital velocity at that altitude | ≈ 1,690 m/s |

**Landing gear limits — exceed any of these and it is a crash:**
- Vertical descent rate at contact: **≤ 3.0 m/s**
- Horizontal velocity at contact: **≤ 1.2 m/s**
- Vehicle tilt at contact: **≤ 12°**
- Touching down with 0 kg propellant remaining: mission failure

For reference, Apollo 11 touched down at roughly 0.5 m/s with about 25 seconds of
propellant margin. That is the standard to aim for.

---

### 4. APOLLO AUTHENTICITY — THE DSKY INTERFACE

Reproduce the look and logic of the Display and Keyboard unit, not the actual AGC code.

**Display layout**
- Three indicator fields at top: `PROG` (2 digits), `VERB` (2 digits), `NOUN` (2 digits)
- Three data registers below: `R1`, `R2`, `R3` — each 5 digits with a sign
- Warning lamp panel: `PROG`, `RESTART`, `GIMBAL LOCK`, `NO ATT`, `VEL`, `ALT`, `UPLINK ACTY`
- Green-on-black electroluminescent styling; fixed-width font

**Verb/Noun combinations to implement**

| Command | Displays |
|---|---|
| V06 N62 | R1 = velocity magnitude (m/s), R2 = altitude rate (m/s), R3 = altitude (m) |
| V16 N68 | Landing monitor: R1 = slant range to target, R2 = time-to-go, R3 = altitude rate |
| V16 N63 | R1 = total velocity, R2 = altitude rate, R3 = altitude |
| V06 N60 | R1 = forward velocity, R2 = altitude rate, R3 = altitude |
| V05 N09 | Display active alarm codes |
| V37 Enn | Change program (e.g. V37 E63 selects P63) |
| V16 N65 | Mission elapsed time (hh:mm:ss across R1/R2/R3) |
| V06 N94 | Propellant remaining and estimated burn time left |

Keys to implement: `VERB`, `NOUN`, `PROG`, digits `0-9`, `+`, `-`, `ENTR`, `CLR`, `KEY REL`,
`RSET`. The `V16` prefix means *monitor* — the display refreshes continuously until
another command is entered.

**Descent programs**

| Program | Name | Behaviour |
|---|---|---|
| **P63** | Braking Phase | High-thrust retrograde burn killing most orbital velocity. Long, mostly steady. |
| **P64** | Approach Phase | Pitch-over so the landing site becomes visible; introduce the Landing Point Designator. |
| **P65** | Automatic Landing | Guidance flies a terminal descent profile automatically. |
| **P66** | Rate of Descent (ROD) | **Manual mode — this is what Armstrong actually used.** Pilot commands attitude; each ROD input adjusts the target descent rate by ±0.3 m/s and the computer holds it. |
| **P67** | Manual Full | Direct throttle and attitude control, no guidance assistance. |

**The 1202 alarm (implement this — it is the best part)**
- Simulate a computation-budget model: each active task consumes cycles per tick.
- If demand exceeds capacity, raise a **1202 EXECUTIVE OVERFLOW** alarm (or **1201 NO CORE
  SETS**), light the PROG lamp, and trigger a software restart that reloads state and
  drops the lowest-priority tasks — exactly as the real AGC's restart protection did.
- Critical guidance must survive the restart. The lesson to encode: the alarm during
  Apollo 11's descent was not a failure — it was the computer correctly shedding load and
  recovering. Print an explanation to the console when it fires.

---

### 5. CONTROLS

| Key | Action |
|---|---|
| `W` / `S` | Increase / decrease DPS throttle (clamped to 10–60 % band or full) |
| `A` / `D` | Yaw / rotate vehicle attitude via RCS |
| `Q` / `E` | Translate laterally using RCS |
| `↑` / `↓` | In P66: adjust commanded rate of descent by ±0.3 m/s |
| `Space` | Toggle between P66 (manual ROD) and P65 (auto landing) |
| `0-9`, `V`, `N`, `Enter` | DSKY keypad entry |
| `R` | Reset simulation |
| `P` | Pause |
| `F1` | Toggle physics debug overlay |

---

### 6. MODULE ARCHITECTURE

Produce clean, separated modules — not one large file:

```
lunar_lander/
├── constants.py        # All physical constants, with units and sources
├── physics.py          # State vector, derivative function, RK4 integrator
├── vehicle.py          # LM model: mass, propellant, thrust, attitude, RCS
├── guidance.py         # P63/P64/P65/P66/P67 program logic, ROD controller
├── agc.py              # Executive scheduler, task priorities, 1202/1201 alarms, restart
├── dsky.py             # DSKY rendering and verb/noun command parsing
├── display.py          # Main render loop: terrain, vehicle, HUD, trajectory trace
├── telemetry.py        # CSV logging of full state at every timestep
├── analysis.py         # Post-flight plots from the telemetry log
└── main.py             # Entry point and simulation loop
```

**Code requirements**
- Type hints throughout; docstrings on every public function.
- Physics functions must be **pure** — state in, derivative out, no side effects. This
  keeps them unit-testable.
- Include `pytest` tests covering: free-fall against the analytical solution, the rocket
  equation against a hand calculation, and energy conservation in an unpowered coast.
- Comment the *physics*, not the syntax. Explain why an equation has the form it has.

---

### 7. BUILD PHASES — IMPLEMENT ONE AT A TIME

Do not build all of these in one pass. Complete and test each phase before moving on.

**Phase 1 — Minimum viable lander**
2D vertical-only motion, constant gravity, constant vehicle mass, fixed thrust that is
either on or off, simple altitude and velocity text readout, crash-or-land verdict.
*Goal: correct integration and a closed loop.*

**Phase 2 — Real propulsion**
Finite propellant, mass depletion coupled to thrust, throttleable DPS within the real
10–60 % band, Isp-based fuel flow, delta-v remaining display, fuel-exhaustion failure.
*Goal: understand why mass fraction dominates rocket design.*

**Phase 3 — Two dimensions and attitude**
Horizontal velocity, vehicle rotation, thrust vectoring along the body axis, RCS
translation, all three touchdown limits enforced, terrain with a designated landing site.
*Goal: realistic control difficulty.*

**Phase 4 — Orbital mechanics**
Start from a real 15.2 km perilune at ~1,690 m/s orbital velocity. Inverse-square gravity.
Implement P63 braking, P64 approach, P66 manual ROD. Add the full descent timeline.
*Goal: a genuine Apollo-profile descent, not a hover puzzle.*

**Phase 5 — The AGC layer**
Full DSKY with verb/noun entry, program switching, warning lamps, the executive task
scheduler, and the 1202/1201 alarm and restart logic.
*Goal: understand how flight software degrades gracefully instead of failing.*

**Phase 6 — Autopilot and estimation**
Add a PID controller for automatic descent-rate hold (P65), and a **Kalman filter** that
estimates altitude and velocity from deliberately noisy simulated radar-altimeter and
IMU measurements. Let the user toggle between raw and filtered state to see the
difference.
*Goal: the single most employable skill in this project — state estimation is the core of
guidance, navigation and control.*

**Phase 7 — Analysis tooling**
Telemetry logging to CSV, post-flight plots (altitude vs time, velocity profile, mass
depletion, thrust history, trajectory ground track), and comparison against the actual
Apollo 11 descent timeline.

---

### 8. LATER EXPANSION — DESIGN FOR IT, DO NOT BUILD IT YET

Structure the code so these can be added, but do not implement them until Phase 7 is
complete and working.

**Ascent and rendezvous**
Ascent Propulsion System (15,600 N, Isp ≈ 311 s), staging away the descent stage,
insertion into lunar orbit, and rendezvous with the Command Module using the
Clohessy-Wiltshire relative-motion equations.

**Rocket configuration builder**
A vehicle-design mode where stages, engines, tanks and payload are selected, and the tool
computes total delta-v via the rocket equation and reports whether a given mission is
achievable. This is where the "rocket building" idea belongs — as an engineering
calculator, not a construction toy.

**Interplanetary mission planner**
Hohmann transfer calculation, launch-window and porkchop-plot generation for Earth-Mars,
patched-conic trajectory approximation, and gravity-assist modelling.

**Propellant depots — the physically correct version**
Model orbital refuelling stations at Lagrange points (L1 and L2 of the Earth-Moon and
Sun-Earth systems) rather than in arbitrary orbits, because these are the locations that
are station-keeping-cheap. Include In-Situ Resource Utilisation: electrolysing lunar or
Martian water ice into hydrogen and oxygen propellant. This is the real, studied answer
to the fuel-station problem.

**Advanced propulsion — real options only**
- **Solar sails:** thrust from photon momentum transfer, roughly 9 μN/m² at 1 AU, falling
  with the inverse square of solar distance. Slow but propellant-free. Japan's IKAROS
  demonstrated this in flight.
- **Nuclear thermal propulsion:** a reactor heats hydrogen propellant to high exhaust
  velocity, giving Isp around 900 s — roughly double chemical rockets. Still expels
  reaction mass; the reactor only supplies the heat.
- **Electric propulsion:** ion and Hall-effect thrusters, Isp of 3,000+ s at very low
  thrust. This is the one place where electromagnetic fields genuinely produce thrust —
  by accelerating ionised propellant, not by pushing against a planet's field.
- **Solar storms and the solar wind:** usable through magnetic sail concepts, but the
  achievable thrust is minute. Model it honestly if included — do not overstate it.

---

### 9. TECHNOLOGY STACK

- **Python 3.11+**
- **pygame** for rendering and input (chosen for simplicity; keep rendering isolated in
  `display.py` so the physics core stays reusable)
- **NumPy** for vector mathematics
- **matplotlib** for post-flight analysis plots
- **pytest** for the test suite

No game engine, no external physics library. The physics is the project.

---

### 10. WHAT I WANT FROM YOU

1. Ask me which phase I am on before writing code.
2. Deliver complete, runnable files for that phase only — no placeholders, no `TODO`
   stubs in the physics.
3. Explain the physics behind each equation as you write it, in comments and in your reply.
4. Where I have made an incorrect physical assumption, correct me directly and explain why.
5. After each phase, tell me what to test manually to confirm it works before I move on.
6. Suggest one specific thing I could measure or plot to deepen my understanding of that
   phase.

Start by asking which phase I want to build, then build it.

## PROMPT ENDS HERE

---

## Notes for you, Kanak (not part of the prompt)

- **Start with Phase 1 only.** Paste sections 1–6 plus Phase 1, and nothing else. A
  working vertical lander in one evening beats a broken orbital simulator in three weeks.
- **The Apollo source you have** — `Comanche055` (Command Module) and `Luminary099`
  (Lunar Module) — is reference material, not code to import. Read
  `THE_LUNAR_LANDING.agc` and the `SERVICER` routines in Luminary099 to see how P63–P66
  were actually structured, then implement your own version in Python.
- **Phase 6 is the one that gets you hired.** Kalman filtering on noisy sensor data is
  exactly what GNC engineers do, and it connects directly to your existing ML background.
- **Keep the telemetry logs.** Being able to show plots of your descent profile next to
  the real Apollo 11 timeline is a far stronger portfolio piece than the simulator alone.
