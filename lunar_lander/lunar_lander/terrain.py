"""
Procedural lunar terrain — Phase 6.

Generates realistic lunar terrain using layered noise functions.
The terrain model provides:
    1. Surface height at any horizontal position
    2. A guaranteed flat landing pad at the target site
    3. Craters, ridges, and boulder fields
    4. Terrain slope for landing safety assessment

WHY THIS MATTERS
────────────────
In Phases 1-5, the LM landed on a perfectly flat surface at y=0.
That's not the Moon. The real Apollo 11 descent path crossed crater
fields, boulders, and slopes that forced Armstrong to manually fly
past the planned landing site to find a safe spot.

Phase 6 changes altitude to mean "height above terrain" rather than
"height above reference sphere." The landing radar measures actual
distance to the ground below, which can be very different from the
inertial altitude the AGC is tracking.

TERRAIN GENERATION
──────────────────
We use layered sine waves (poor man's Perlin noise) with different
frequencies and amplitudes:
    - Large features (λ=2000m, A=80m): Major ridges and basins
    - Medium features (λ=500m, A=30m): Hills and depressions
    - Small features (λ=100m, A=10m): Boulders and rubble
    - Craters: Circular depressions at fixed locations

The landing pad is a flat zone of ±50 m around the landing site.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ── Terrain generation seed ──
TERRAIN_SEED: int = 11  # Apollo 11


@dataclass(frozen=True)
class TerrainLayer:
    """A single frequency layer of the terrain heightmap.

    Terrain is built by summing multiple sine waves at different
    frequencies (wavelengths) and amplitudes. This is a simplified
    version of fractal noise / fBm (fractional Brownian motion).

    Attributes:
        wavelength: Spatial period [m]. Larger = broader hills.
        amplitude:  Max height variation [m]. Larger = taller features.
        phase:      Phase offset [rad]. Randomized for variety.
    """
    wavelength: float
    amplitude: float
    phase: float


@dataclass
class Crater:
    """A circular crater depression.

    Attributes:
        x:      Center position [m].
        radius: Crater radius [m].
        depth:  Maximum depth [m] (positive = downward).
        rim:    Rim height above surrounding terrain [m].
    """
    x: float
    radius: float
    depth: float
    rim: float


class LunarTerrain:
    """Procedural lunar terrain generator.

    Provides a continuous height function h(x) that returns the surface
    elevation at any horizontal position. Heights are relative to the
    reference sphere (R_moon), so the "ground" is at h(x) rather than 0.

    The terrain is deterministic (seeded) so it's the same every run.

    Usage:
        terrain = LunarTerrain(landing_site_x=0.0)
        height = terrain.height_at(x_position)
        slope = terrain.slope_at(x_position)
    """

    def __init__(
        self,
        landing_site_x: float = 0.0,
        pad_half_width: float = 50.0,
        seed: int = TERRAIN_SEED,
    ) -> None:
        """Initialize terrain with deterministic features.

        Args:
            landing_site_x: Center of the flat landing pad [m].
            pad_half_width: Half-width of the flat landing zone [m].
            seed:           Random seed for reproducibility.
        """
        self.landing_site_x = landing_site_x
        self.pad_half_width = pad_half_width

        rng = random.Random(seed)

        # ── Terrain layers ──
        # Large-scale features: maria ridges, basin edges
        # Medium: hills, old crater rims
        # Small: boulders, rubble fields
        self.layers: list[TerrainLayer] = [
            TerrainLayer(wavelength=3000.0, amplitude=60.0,
                         phase=rng.uniform(0, 2 * math.pi)),
            TerrainLayer(wavelength=1200.0, amplitude=35.0,
                         phase=rng.uniform(0, 2 * math.pi)),
            TerrainLayer(wavelength=500.0,  amplitude=18.0,
                         phase=rng.uniform(0, 2 * math.pi)),
            TerrainLayer(wavelength=200.0,  amplitude=8.0,
                         phase=rng.uniform(0, 2 * math.pi)),
            TerrainLayer(wavelength=80.0,   amplitude=3.0,
                         phase=rng.uniform(0, 2 * math.pi)),
            TerrainLayer(wavelength=30.0,   amplitude=1.2,
                         phase=rng.uniform(0, 2 * math.pi)),
        ]

        # ── Craters ──
        # Place craters at random positions, avoiding the landing pad
        self.craters: list[Crater] = []
        for _ in range(25):
            cx = rng.uniform(-15000, 60000)
            # Don't put craters on the landing pad
            if abs(cx - landing_site_x) < pad_half_width + 80:
                continue
            radius = rng.uniform(20, 200)
            depth = radius * rng.uniform(0.15, 0.4)
            rim = depth * rng.uniform(0.08, 0.15)
            self.craters.append(Crater(cx, radius, depth, rim))

        # ── Cache the pad height ──
        # The pad sits at the average terrain height at the landing site
        # but is perfectly flat
        self._pad_height = self._raw_height(landing_site_x)

    def _raw_height(self, x: float) -> float:
        """Compute terrain height WITHOUT the landing pad flattening.

        Sums all terrain layers and crater depressions.

        Args:
            x: Horizontal position [m].

        Returns:
            Terrain height [m] above the reference sphere.
        """
        h = 0.0

        # Sum sine layers
        for layer in self.layers:
            h += layer.amplitude * math.sin(
                2.0 * math.pi * x / layer.wavelength + layer.phase
            )

        # Apply craters
        for crater in self.craters:
            dist = abs(x - crater.x)
            if dist < crater.radius * 1.5:
                # Inside crater bowl: parabolic depression
                if dist < crater.radius:
                    t = dist / crater.radius
                    h -= crater.depth * (1.0 - t * t)
                # Crater rim: slight elevation
                elif dist < crater.radius * 1.3:
                    t = (dist - crater.radius) / (crater.radius * 0.3)
                    h += crater.rim * (1.0 - t)

        return h

    def height_at(self, x: float) -> float:
        """Get terrain height at position x.

        Returns the height of the terrain surface at horizontal position x.
        The landing pad area is guaranteed flat.

        Args:
            x: Horizontal position [m].

        Returns:
            Terrain surface height [m] relative to reference sphere.
            Positive = above reference, negative = below reference.
        """
        dist_to_pad = abs(x - self.landing_site_x)

        if dist_to_pad <= self.pad_half_width:
            # On the landing pad — perfectly flat
            return self._pad_height

        # Smooth transition zone at pad edges (20m blend)
        blend_width = 20.0
        if dist_to_pad < self.pad_half_width + blend_width:
            t = (dist_to_pad - self.pad_half_width) / blend_width
            # Smooth interpolation (smoothstep)
            t = t * t * (3.0 - 2.0 * t)
            raw = self._raw_height(x)
            return self._pad_height * (1.0 - t) + raw * t

        return self._raw_height(x)

    def slope_at(self, x: float, dx: float = 1.0) -> float:
        """Compute terrain slope at position x.

        Uses central difference: slope ≈ (h(x+dx) - h(x-dx)) / (2·dx)

        LANDING SAFETY
        ──────────────
        The Apollo LM could land on slopes up to about 12°. Steeper
        slopes risked the vehicle tipping over. During the real mission,
        Armstrong flew past the planned site because he saw boulders
        and slopes that would have been dangerous.

        Args:
            x:  Horizontal position [m].
            dx: Finite difference step [m].

        Returns:
            Slope angle [rad]. Positive = uphill to the right.
        """
        dh = self.height_at(x + dx) - self.height_at(x - dx)
        return math.atan2(dh, 2.0 * dx)

    def is_safe_landing(self, x: float, max_slope_deg: float = 6.0) -> bool:
        """Check if a position is safe for landing.

        Evaluates terrain slope over the landing gear footprint (~9m).

        Args:
            x: Horizontal position [m].
            max_slope_deg: Maximum safe slope [degrees].

        Returns:
            True if the terrain is safe for landing at this position.
        """
        for offset in [-4.5, -2.0, 0.0, 2.0, 4.5]:
            slope_deg = abs(math.degrees(self.slope_at(x + offset)))
            if slope_deg > max_slope_deg:
                return False
        return True

    def get_terrain_profile(
        self,
        x_center: float,
        width: float,
        num_points: int = 200,
    ) -> list[tuple[float, float]]:
        """Get a terrain height profile for rendering.

        Args:
            x_center:   Center of the profile [m].
            width:      Total width [m].
            num_points: Number of sample points.

        Returns:
            List of (x, height) tuples.
        """
        x_start = x_center - width / 2.0
        dx = width / num_points
        return [
            (x_start + i * dx, self.height_at(x_start + i * dx))
            for i in range(num_points + 1)
        ]
