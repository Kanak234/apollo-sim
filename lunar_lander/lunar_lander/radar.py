"""
Landing Radar simulation — Phase 6.

Simulates the Apollo LM's landing radar, which measured:
    1. Altitude above the terrain (range)
    2. Three velocity components (Doppler)

THE LANDING RADAR
─────────────────
The real Apollo LM had a four-beam radar:
    - Beam 1: range (altitude above terrain)
    - Beams 2-4: three Doppler beams for 3D velocity

The radar was mounted on the descent stage, pointing downward.
It had to be deployed (rotated to the landing position) before it
could be used.

Key characteristics:
    - Maximum range: ~12,000 m (40,000 ft)
    - Minimum range: ~3 m (10 ft)
    - Range accuracy: ±0.4% of reading
    - Velocity accuracy: ±0.15 m/s per axis
    - Update rate: ~2 Hz for range, ~5 Hz for velocity
    - Lock time: ~1-2 seconds after deployment

RADAR vs INERTIAL
─────────────────
Before radar lock, the AGC knows altitude only from its inertial
measurement unit (IMU). The IMU integrates accelerations to estimate
position, but errors accumulate over time. By the time of PDI
(~8 minutes of thrust), the IMU altitude can be off by hundreds
of meters.

When the radar locks on, the AGC has a direct measurement of actual
altitude above the terrain below. The difference between IMU and
radar altitude is called the "radar residual." If the residual is
too large, the AGC triggers an alarm (1202) and the crew must decide
whether to accept the radar data.

RADAR DATA GOOD flag
────────────────────
The AGC set a "radar data good" flag when:
    1. The radar had locked on (signal quality above threshold)
    2. The radar readings were consistent (no wild jumps)
    3. The altitude was below 12 km (radar range limit)

When the flag was set, the AGC blended radar data with IMU data
to get a better altitude estimate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from terrain import LunarTerrain


# ── Radar specifications ──
RADAR_MAX_RANGE: float = 12_000.0     # Maximum altitude for lock [m]
RADAR_MIN_RANGE: float = 3.0          # Minimum range [m]
RADAR_RANGE_ERROR: float = 0.004      # ±0.4% of reading
RADAR_VEL_ERROR: float = 0.15         # ±0.15 m/s per axis
RADAR_LOCK_TIME: float = 1.5          # Time to acquire lock [s]
RADAR_UPDATE_RATE: float = 2.0        # Range updates per second [Hz]


@dataclass
class RadarReading:
    """A single radar measurement.

    Attributes:
        range_valid:    Whether range data is valid.
        velocity_valid: Whether velocity data is valid.
        range_m:        Measured altitude above terrain [m].
        velocity_x:     Measured horizontal velocity [m/s].
        velocity_y:     Measured vertical velocity [m/s].
        terrain_height: Height of the terrain directly below [m].
        true_agl:       True altitude above ground level [m] (for debug).
        data_good:      Whether the AGC should trust this data.
    """
    range_valid: bool = False
    velocity_valid: bool = False
    range_m: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    terrain_height: float = 0.0
    true_agl: float = 0.0
    data_good: bool = False


class LandingRadar:
    """Simulates the Apollo LM landing radar.

    The radar provides altitude-above-terrain and velocity measurements,
    with realistic noise, lock behavior, and range limits.

    MEASUREMENT MODEL
    ─────────────────
    Range: radar_alt = true_agl + noise
        where noise ~ N(0, σ), σ = RADAR_RANGE_ERROR × true_agl

    Velocity: radar_vel = true_vel + noise
        where noise ~ N(0, RADAR_VEL_ERROR) per axis

    The noise model is Gaussian (normal distribution). In reality,
    the radar used frequency-modulated continuous wave (FMCW) for range
    and Doppler shift for velocity, both of which have approximately
    Gaussian noise characteristics.
    """

    def __init__(self, terrain: LunarTerrain, seed: int = 42) -> None:
        """Initialize the landing radar.

        Args:
            terrain: The terrain model to measure altitude against.
            seed:    Random seed for measurement noise.
        """
        self.terrain = terrain
        self._rng = random.Random(seed)

        # State
        self._locked: bool = False
        self._lock_timer: float = 0.0
        self._last_update_time: float = -1.0  # sentinel: not yet updated
        self._last_reading = RadarReading()

        # Track measurement consistency for data quality
        self._prev_range: float = 0.0
        self._range_jump_count: int = 0

    @property
    def is_locked(self) -> bool:
        """Whether the radar has acquired a lock on the terrain."""
        return self._locked

    def update(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        time: float,
    ) -> RadarReading:
        """Update radar state and get current reading.

        The radar goes through these states:
            1. OUT OF RANGE: altitude > 12 km → no lock possible
            2. ACQUIRING: entered range, waiting for lock timer
            3. LOCKED: valid measurements with noise

        Args:
            x:    Vehicle horizontal position [m].
            y:    Vehicle altitude above reference [m].
            vx:   Vehicle horizontal velocity [m/s].
            vy:   Vehicle vertical velocity [m/s].
            time: Current mission elapsed time [s].

        Returns:
            RadarReading with current measurements.
        """
        terrain_h = self.terrain.height_at(x)
        true_agl = y - terrain_h

        reading = RadarReading()
        reading.terrain_height = terrain_h
        reading.true_agl = true_agl

        # ── Check if in range ──
        if true_agl > RADAR_MAX_RANGE or true_agl < RADAR_MIN_RANGE:
            self._locked = False
            self._lock_timer = 0.0
            self._last_reading = reading
            return reading

        # ── Lock acquisition ──
        if not self._locked:
            if self._last_update_time >= 0:
                self._lock_timer += time - self._last_update_time
            if self._lock_timer >= RADAR_LOCK_TIME:
                self._locked = True
                self._prev_range = true_agl
                self._last_update_time = time
                print(f"  ► RADAR LOCK — altitude {true_agl:.0f} m AGL")
            else:
                self._last_update_time = time
                self._last_reading = reading
                return reading

        # ── Generate noisy measurements ──
        # Range measurement
        range_sigma = RADAR_RANGE_ERROR * true_agl
        range_noise = self._rng.gauss(0.0, max(range_sigma, 0.5))
        reading.range_m = max(0.0, true_agl + range_noise)
        reading.range_valid = True

        # Velocity measurements
        vx_noise = self._rng.gauss(0.0, RADAR_VEL_ERROR)
        vy_noise = self._rng.gauss(0.0, RADAR_VEL_ERROR)
        reading.velocity_x = vx + vx_noise
        reading.velocity_y = vy + vy_noise
        reading.velocity_valid = True

        # ── Data quality check ──
        # If range jumps by more than 20% between readings, flag as suspect
        if self._prev_range > 0:
            jump = abs(reading.range_m - self._prev_range) / self._prev_range
            if jump > 0.2:
                self._range_jump_count += 1
            else:
                self._range_jump_count = max(0, self._range_jump_count - 1)

        reading.data_good = (
            reading.range_valid
            and reading.velocity_valid
            and self._range_jump_count < 3
        )

        self._prev_range = reading.range_m
        self._last_update_time = time
        self._last_reading = reading
        return reading

    def get_last_reading(self) -> RadarReading:
        """Get the most recent radar reading without updating."""
        return self._last_reading

    def altitude_agl(self, x: float, y: float) -> float:
        """Quick true altitude above ground level (no noise).

        Used for collision detection — not available to the pilot!

        Args:
            x: Horizontal position [m].
            y: Altitude above reference [m].

        Returns:
            True altitude above terrain [m].
        """
        return y - self.terrain.height_at(x)
