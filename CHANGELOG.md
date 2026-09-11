# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-12

### Added
- **Production Hardening (C1–C9):** Complete production readiness suite meeting criteria C1 through C9.
- **Unified Package & CLI:** Added `apollo-sim` console script with CLI arguments (`--mission {apollo11,mars}`, `--plots`, `--version`).
- **Automated CI Matrix:** Multi-version Python matrix (`[3.10, 3.11, 3.12, 3.13]`) testing, ruff linting, test coverage gating, and Docker container verification.
- **CodeQL Workflow:** Automated static analysis and security scanning workflow for Python (`.github/workflows/codeql.yml`).
- **Automated Release Workflow:** GitHub Actions release pipeline building source distributions and wheels with SHA256 checksum generation (`.github/workflows/release.yml`).
- **Dependabot Integration:** Automated weekly dependency audits for pip, GitHub Actions, and Docker (`.github/dependabot.yml`).
- **Multi-Stage Dockerfile:** Reproducible unprivileged container build running as non-root user `apollo` (UID 10001) with automated smoke test entrypoint.
- **Mission Test Suite:** Added `apollo_sim/tests/test_missions.py` and `apollo_sim/tests/test_cli.py`, expanding the test suite to 189 tests with >92% statement coverage and `--cov-fail-under=85` gate.
- **Security Policy:** Threat model covering astrodynamics numerical integrity, singularity mitigation, and container sandboxing (`SECURITY.md`).

### Fixed
- Setuptools package discovery in `pyproject.toml` resolving multi-package layout (`apollo_sim` and `lunar_lander`).
- Import resolution in `apollo_sim/cli.py` ensuring internal package modules resolve seamlessly when invoked from any working directory.
