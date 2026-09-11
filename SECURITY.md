# Security Policy

## Supported Versions

Security updates are provided for the latest production release of `apollo-sim`.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

---

## Reporting a Vulnerability

If you identify a security or numerical integrity vulnerability in `apollo-sim`, please disclose it responsibly. Do not submit public issues for security vulnerabilities.

### Reporting Steps
1. Email security reports to **kanakprabhakar2@gmail.com** or create a private advisory at `https://github.com/Kanak234/apollo-sim/security/advisories/new`.
2. Provide details including simulation configuration, parameters triggering the issue, and expected vs observed behavior.
3. Vulnerability reports receive an initial response within 48 hours.

---

## Threat Model & Numerical Integrity

`apollo-sim` is an astrodynamics and flight mechanics simulation library. The principal threat model concerns numerical stability, calculation integrity, and container boundaries:

### 1. Numerical Stability & Singularity Mitigation
- Planetary and lunar gravity fields model central bodies with physical radius thresholds ($r > R_{surface}$) to prevent zero-distance division singularities ($1/r^2 \to \infty$).
- Integrator time steps ($\Delta t$) are bounded to prevent energy divergence and artificial hyperbolic escape during close periapsis passages.
- Atmospheric entry calculations clamp densities and Mach numbers to prevent NaN propagation during extreme stagnation temperatures.

### 2. Container Sandboxing
- Docker images run under an unprivileged user (`apollo`, UID/GID 10001) without sudo or root privileges.
- Read-only simulation execution guarantees that file system tampering cannot occur within container environments.

### 3. Dependency Scope
- The core simulation library runs headless with `numpy` as its sole runtime dependency, eliminating attack surface associated with web frameworks or network listeners.
