# Apollo-to-Mars Simulator

Two packages, one project:

```
lunar_lander/    Phases 1-6  — the interactive pygame game (your build,
                 with the orbital-gravity and attitude-hold fixes applied)
apollo_sim/      Phases 7-15 — headless, tested simulation library +
                 mission drivers (no pygame; physics and reports)
```

## Quick start

```bash
# the game (needs pygame)
cd lunar_lander && python3 main.py

# the test suites
cd lunar_lander && python3 -m pytest tests/      # 133 tests
cd apollo_sim   && python3 -m pytest tests/      # 43 tests

# full missions
cd apollo_sim
python3 run_apollo11.py --plots     # launch -> splashdown, dv ledger
python3 run_mars.py --plots         # windows, porkchop, EDL, ISRU, depots
```

## What each phase is, and where it lives

| Phase | Content                              | Where                          |
|------:|--------------------------------------|--------------------------------|
| 1-6   | Interactive LM descent (game)        | `lunar_lander/`                |
| 7     | Telemetry/analysis                   | drivers print MET reports + PNGs |
| 8     | Saturn V ascent, losses ledger       | `apollo_sim/ascent.py`         |
| 9     | TLI + patched-conic translunar       | `apollo_sim/transfer.py`       |
| 10    | LOI-1 / LOI-2                        | `apollo_sim/transfer.py`       |
| 11    | Lunar ascent + CW rendezvous         | `apollo_sim/rendezvous.py`     |
| 12    | Entry corridor, heating, chutes      | `apollo_sim/entry.py`          |
| 13    | Depots, boil-off, architecture trade | `apollo_sim/marsops.py`        |
| 14    | Lambert, windows, porkchop           | `apollo_sim/marstransfer.py`   |
| 15    | Mars EDL + ISRU                      | `apollo_sim/marsops.py`        |
| —     | Automated powered descent (P63-style)| `apollo_sim/descent_auto.py`   |

## Verified against known values (all in the test suites)

| Quantity                     | Simulated | Reference        |
|------------------------------|-----------|------------------|
| LEO insertion ideal delta-v  | ~9,650    | ~9,400 m/s       |
| TLI                          | 3,177     | 3,050-3,200 m/s  |
| LOI-1                        | 913       | 889 m/s          |
| Powered descent              | 2,127     | ~2,100 m/s       |
| Lunar ascent                 | 1,885     | 1,850-1,950 m/s  |
| Entry corridor               | -5.5..-7.3 deg lands | -6.5 +/- ~1 deg |
| Splashdown speed             | 8.3       | ~8.5 m/s         |
| Synodic period               | 780.0     | 779.9 days       |
| Hohmann TOF Earth->Mars      | 258.9     | ~259 days        |
| TMI from LEO                 | 3,616     | ~3,600 m/s       |
| Porkchop minimum vs Hohmann  | 5,709 vs 5,708 m/s | agree      |

## Simplifications (deliberate, documented)

- 2D, coplanar everywhere. No inclinations, no plane changes.
- Patched conics; no n-body integration, no free-return computation.
- Circular-coplanar planetary ephemerides for Mars work.
- Insertion/LOI/TLI/TEI burns treated impulsively where noted, always
  with rocket-equation mass accounting.
- Exponential atmospheres.
- Phases 8-15 have no interactive display. The physics core is what a
  display would sit on; building the UI is UI work, not physics work.
- "Tested" means: checked against analytical two-body results and the
  historical Apollo values above. This is an educational simulator,
  not certified flight software — nothing on a laptop is.

## The one bug worth remembering

The original 2D engine applied gravity as a constant -y force. Flat
world. All 120 of its tests passed, because the tests assumed the same
flat world. A circular orbit hit the ground in 138 seconds.

Tests prove self-consistency. Only independent physics proves
correctness. `lunar_lander/tests/test_orbital.py` now pins the truth.

## For the game: central_gravity

The engine fix ships in `lunar_lander/physics.py` behind
`central_gravity=True` (default off in the game loop, because the
Phase 1-6 guidance was tuned against the flat model). The automated
descent in `apollo_sim/descent_auto.py` demonstrates the full descent
working under correct orbital gravity — port its braking law into the
game's autopilot when you want the game itself on the round Moon.
