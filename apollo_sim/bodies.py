"""Celestial bodies and reference constants (SI units).

Sources: NASA planetary fact sheets (NSSDCA), IAU 2015 nominal values.
Atmospheres are simple exponential models: rho(h) = rho0 * exp(-h/H).
That is the standard first-order model; it is what makes max-Q, entry
corridors and Mars EDL behave qualitatively correctly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    name: str
    mu: float                 # gravitational parameter [m^3/s^2]
    radius: float             # mean radius [m]
    atm_rho0: float = 0.0     # sea-level density [kg/m^3]
    atm_scale_h: float = 1.0  # scale height [m]
    atm_top: float = 0.0      # altitude above which atmosphere is ignored [m]
    soi: float = 0.0          # sphere-of-influence radius [m]

    def gravity(self, r: float) -> float:
        """Gravitational acceleration magnitude at radius r from centre."""
        return self.mu / (r * r)

    def density(self, altitude: float) -> float:
        """Atmospheric density at altitude [m] above the surface."""
        if self.atm_rho0 <= 0.0 or altitude > self.atm_top or altitude < 0.0:
            return 0.0 if altitude > 0.0 else self.atm_rho0
        import math
        return self.atm_rho0 * math.exp(-altitude / self.atm_scale_h)


G0 = 9.80665  # standard gravity, defines Isp [m/s^2]

EARTH = Body("Earth", mu=3.986004418e14, radius=6_371_000.0,
             atm_rho0=1.225, atm_scale_h=8_500.0, atm_top=140_000.0,
             soi=9.24e8)

MOON = Body("Moon", mu=4.9028695e12, radius=1_737_400.0, soi=6.61e7)

MARS = Body("Mars", mu=4.282837e13, radius=3_389_500.0,
            atm_rho0=0.020, atm_scale_h=11_100.0, atm_top=125_000.0,
            soi=5.77e8)

SUN = Body("Sun", mu=1.32712440018e20, radius=6.957e8)

EARTH_MOON_DIST = 3.844e8       # m, mean
SUN_EARTH_DIST = 1.496e11       # m (1 AU)
SUN_MARS_DIST = 2.2794e11       # m, mean
EARTH_ROT_SPEED_KSC = 408.0     # m/s eastward at 28.5 deg latitude
