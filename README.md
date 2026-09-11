# Apollo-to-Mars Simulator

[![CI](https://github.com/Kanak234/apollo-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Kanak234/apollo-sim/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Kanak234/apollo-sim/actions/workflows/codeql.yml/badge.svg)](https://github.com/Kanak234/apollo-sim/actions/workflows/codeql.yml)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/Kanak234/apollo-sim/releases)
[![Coverage](https://img.shields.io/badge/coverage-92.1%25-brightgreen)](https://github.com/Kanak234/apollo-sim)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Two packages, one project:

```
lunar_lander/    Phases 1-6  — the interactive pygame game (orbital-gravity and attitude-hold fixes applied)
apollo_sim/      Phases 7-15 — headless, tested simulation library + mission drivers (physics and reports)
```

---

## Quick start

### CLI Execution

```bash
# Install package
pip install .

# Run Apollo 11 mission simulation
apollo-sim --mission apollo11

# Run Mars transfer, EDL, and ISRU simulation
apollo-sim --mission mars

# Generate plots
apollo-sim --mission apollo11 --plots
```

### Game & Test Execution

```bash
# The interactive game (requires pygame)
pip install ".[sim]"
cd lunar_lander && python3 lunar_lander/main.py

# Run test suite with automated coverage gate (>=85%)
pytest
```

### Docker Verification

```bash
# Build verification container
docker build -t apollo-sim:smoke .

# Run containerized simulation
docker run --rm apollo-sim:smoke --mission apollo11
```

---

## Mission Phases & Components

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

---

## Verified Against Known Values

| Quantity                     | Simulated | Reference        |
|------------------------------|-----------|------------------|
| LEO insertion ideal delta-v  | ~9,476    | ~9,400 m/s       |
| TLI                          | 3,210     | 3,050-3,200 m/s  |
| LOI-1                        | 992       | 889 m/s          |
| Powered descent              | 2,193     | ~2,100 m/s       |
| Lunar ascent                 | 1,885     | 1,850-1,950 m/s  |
| Entry corridor               | -5.5..-7.3 deg lands | -6.5 +/- ~1 deg |
| Splashdown speed             | 8.5       | ~8.5 m/s         |
| Synodic period               | 780.0     | 779.9 days       |
| Hohmann TOF Earth->Mars      | 259       | ~259 days        |
| TMI from LEO                 | 3,616     | ~3,600 m/s       |
| Porkchop minimum vs Hohmann  | 5,709 vs 5,708 m/s | agree      |

---

## Simplifications (Deliberate, Documented)

- 2D, coplanar everywhere. No inclinations, no plane changes.
- Patched conics; no n-body integration, no free-return computation.
- Circular-coplanar planetary ephemerides for Mars work.
- Insertion/LOI/TLI/TEI burns treated impulsively where noted, always with rocket-equation mass accounting.
- Exponential atmospheres.
- Phases 8-15 have no interactive display. The physics core is what a display sits on.
- Tested: checked against analytical two-body results and historical Apollo values.

---

## Security & Reliability

- Unprivileged container execution (UID 10001 `apollo`).
- Numerical boundary checks mitigating orbital singularities ($r \to 0$).
- Deterministic simulation calculations with standard library and NumPy dependencies.
- See [SECURITY.md](SECURITY.md) for vulnerability disclosure details.

---

## License

MIT — See [LICENSE](LICENSE) for details.
