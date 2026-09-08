"""
Lunar Module vehicle model — Phase 2: Real Propulsion.

This module encapsulates the LM's mass, propulsion, and throttle state.

Phase 2 adds:
────────────
• Propellant tracking — 8,200 kg of usable propellant that depletes
• Throttle control — 10-60% continuous range plus 100% fixed full thrust
• The "throttle bucket" — the forbidden 60-100% zone
• Mass flow rate — how fast propellant is consumed at current throttle
• Delta-v remaining — Tsiolkovsky equation applied to current state
• Burn time remaining — seconds of propellant at current throttle
• Fuel exhaustion — engine cuts off when propellant reaches zero

KEY INSIGHT THIS PHASE TEACHES:
As propellant burns and mass drops from 15,100 → 6,900 kg, the thrust
acceleration (F/m) rises from 2.98 → 6.53 m/s². Same engine, same throttle,
but 2.2× more responsive. This is why pilots reported the LM feeling
"twitchy" near touchdown — you have to back off the throttle as you get
lighter, which is counter-intuitive when you're also trying to slow down.
"""

from __future__ import annotations

import math

from constants import (
    LM_MASS_PDI,
    DPS_MAX_THRUST,
    DPS_ISP,
    DPS_THROTTLE_MIN,
    DPS_THROTTLE_MAX_CONTINUOUS,
    DPS_THROTTLE_FULL,
    PROPELLANT_MASS,
    G0,
    G_MOON,
)


# Throttle step size for keyboard control.
# The real TTCA (Thrust/Translation Controller Assembly) was a continuous
# lever, but keyboard input requires discrete steps. 5% gives 11 positions
# in the continuous range (10, 15, 20, ..., 55, 60) plus off and full.
THROTTLE_STEP: float = 0.05


class LunarModule:
    """Model of the Apollo Lunar Module descent stage.

    Tracks the vehicle's propulsion state: mass, propellant, throttle,
    and fuel status. Provides computed properties for thrust force,
    acceleration, mass flow rate, delta-v, and burn time.

    Attributes:
        propellant:     Remaining usable propellant [kg].
        throttle:       Current throttle setting [0.0 - 1.0].
        fuel_exhausted: True if propellant has been fully consumed.
    """

    def __init__(self) -> None:
        """Initialize the LM at Powered Descent Initiation configuration.

        At PDI:
            Total mass:  15,100 kg
            Propellant:   8,200 kg (usable)
            Dry mass:     6,900 kg (structure + ascent stage + crew)
            Throttle:     off (0.0)
        """
        self._dry_mass: float = LM_MASS_PDI - PROPELLANT_MASS
        self.propellant: float = PROPELLANT_MASS
        self.throttle: float = 0.0
        self.fuel_exhausted: bool = False
        self._rcs_command: int = 0  # Phase 3: RCS attitude command

    # ════════════════════════════════════════════════════════════════
    # MASS PROPERTIES
    # ════════════════════════════════════════════════════════════════

    @property
    def mass(self) -> float:
        """Current total vehicle mass [kg].

        mass = dry_mass + propellant_remaining

        At PDI:    15,100 kg (full propellant)
        At empty:   6,900 kg (dry mass only)

        As mass decreases, thrust acceleration (F/m) INCREASES.
        This is the core variable-mass dynamic.
        """
        return self._dry_mass + self.propellant

    @property
    def dry_mass(self) -> float:
        """Vehicle mass with zero propellant [kg].

        This is the structural mass: descent stage hardware, ascent stage,
        crew, consumables, science payload. It does not change during the
        descent (staging happens during ascent, Phase 12).

        dry_mass = 15,100 − 8,200 = 6,900 kg
        """
        return self._dry_mass

    @property
    def propellant_fraction(self) -> float:
        """Fraction of propellant remaining [0.0 - 1.0].

        1.0 = full tank (8,200 kg)
        0.0 = empty
        """
        if PROPELLANT_MASS <= 0:
            return 0.0
        return self.propellant / PROPELLANT_MASS

    # ════════════════════════════════════════════════════════════════
    # THROTTLE CONTROL
    # ════════════════════════════════════════════════════════════════

    @property
    def engine_on(self) -> bool:
        """Whether the engine is currently producing thrust."""
        return self.throttle > 0.0 and not self.fuel_exhausted

    def toggle_engine(self) -> None:
        """Toggle engine on/off.

        When toggling ON: sets throttle to the last-used setting,
        or to minimum (10%) if no previous setting.
        When toggling OFF: sets throttle to 0.
        """
        if self.fuel_exhausted:
            return
        if self.throttle > 0.0:
            self._last_throttle = self.throttle
            self.throttle = 0.0
        else:
            self.throttle = getattr(self, '_last_throttle', DPS_THROTTLE_MIN)

    def set_engine(self, on: bool) -> None:
        """Explicitly set engine state. Used for forced shutdown."""
        if not on:
            if self.throttle > 0.0:
                self._last_throttle = self.throttle
            self.throttle = 0.0
        elif not self.fuel_exhausted:
            self.throttle = getattr(self, '_last_throttle', DPS_THROTTLE_MIN)

    def throttle_up(self) -> None:
        """Increase throttle by one step, respecting the throttle bucket.

        The DPS throttle has three zones:
            OFF:        throttle = 0.0
            Continuous: throttle ∈ [0.10, 0.60]  (fine control)
            Full:       throttle = 1.00           (fixed full thrust)

        The "throttle bucket" between 0.60 and 1.00 is FORBIDDEN because
        combustion becomes unstable in that range. The injector flow
        pattern breaks down, causing pressure oscillations that could
        damage the engine or cause an explosion.

        So pressing UP from 0.60 jumps directly to 1.00.

        Step sequence: off → 0.10 → 0.15 → ... → 0.55 → 0.60 → 1.00
        """
        if self.fuel_exhausted:
            return

        if self.throttle == 0.0:
            # Engine off → minimum thrust
            self.throttle = DPS_THROTTLE_MIN
        elif self.throttle < DPS_THROTTLE_MAX_CONTINUOUS - 0.001:
            # In continuous range → step up
            self.throttle = min(
                round(self.throttle + THROTTLE_STEP, 2),
                DPS_THROTTLE_MAX_CONTINUOUS,
            )
        elif self.throttle < DPS_THROTTLE_FULL - 0.001:
            # At max continuous → jump over bucket to full
            self.throttle = DPS_THROTTLE_FULL
        # else: already at full, do nothing

    def throttle_down(self) -> None:
        """Decrease throttle by one step, respecting the throttle bucket.

        Pressing DOWN from 1.00 drops to 0.60 (skipping the bucket).
        Pressing DOWN from 0.10 turns the engine off.

        Step sequence: 1.00 → 0.60 → 0.55 → ... → 0.15 → 0.10 → off
        """
        if self.throttle >= DPS_THROTTLE_FULL - 0.001:
            # Full thrust → drop to max continuous
            self.throttle = DPS_THROTTLE_MAX_CONTINUOUS
        elif self.throttle > DPS_THROTTLE_MIN + 0.001:
            # In continuous range → step down
            self.throttle = max(
                round(self.throttle - THROTTLE_STEP, 2),
                DPS_THROTTLE_MIN,
            )
        elif self.throttle > 0.001:
            # At minimum → engine off
            self.throttle = 0.0
        # else: already off, do nothing

    # ════════════════════════════════════════════════════════════════
    # THRUST PROPERTIES
    # ════════════════════════════════════════════════════════════════

    @property
    def thrust_force(self) -> float:
        """Current thrust force [N].

        thrust = throttle × DPS_MAX_THRUST

        Examples:
            throttle=0.10 → 4,504 N   (minimum continuous)
            throttle=0.30 → 13,512 N  (typical hover-ish)
            throttle=0.60 → 27,024 N  (max continuous)
            throttle=1.00 → 45,040 N  (fixed full thrust, braking)
        """
        if self.fuel_exhausted or self.throttle <= 0.0:
            return 0.0
        return self.throttle * DPS_MAX_THRUST

    @property
    def thrust_acceleration(self) -> float:
        """Current thrust acceleration [m/s²] = F / m.

        This INCREASES as propellant burns:
            At PDI (m=15,100): F/m = 45,040/15,100 = 2.98 m/s²
            Half fuel (m=11,000): F/m = 45,040/11,000 = 4.09 m/s²
            Near empty (m=7,000): F/m = 45,040/7,000 = 6.43 m/s²

        This 2.2× increase is what makes the LM feel "twitchy" at
        low fuel — the same throttle setting produces much more
        acceleration. Pilots had to progressively reduce throttle
        to maintain the same descent rate.
        """
        m = self.mass
        if m <= 0.0:
            return 0.0
        return self.thrust_force / m

    @property
    def twr(self) -> float:
        """Thrust-to-weight ratio on the lunar surface.

        TWR = F / (m × g_moon)

        TWR > 1.0 means the vehicle can hover and ascend.
        TWR < 1.0 means it cannot overcome gravity at current throttle.

        At full thrust:
            PDI:   45,040 / (15,100 × 1.625) = 1.84
            Empty: 45,040 / (6,900 × 1.625)  = 4.02
        """
        m = self.mass
        if m <= 0.0:
            return 0.0
        return self.thrust_force / (m * G_MOON)

    # ════════════════════════════════════════════════════════════════
    # PROPELLANT / FUEL PROPERTIES
    # ════════════════════════════════════════════════════════════════

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity v_e = Isp × g0 [m/s].

        This is the physically meaningful measure of engine efficiency.
        For the DPS: v_e = 311 × 9.80665 ≈ 3,050 m/s.

        It means each kilogram of propellant leaves the nozzle at
        3,050 m/s. By Newton's third law, this imparts 3,050 N·s
        of impulse to the vehicle per kg of fuel burned.
        """
        return DPS_ISP * G0

    @property
    def mass_flow_rate(self) -> float:
        """Current propellant consumption rate [kg/s].

        ṁ = F / v_e = F / (Isp × g0)

        Full thrust:  45,040 / 3,050 ≈ 14.77 kg/s
        10% throttle:  4,504 / 3,050 ≈  1.48 kg/s

        At full thrust, the 8,200 kg supply lasts about 555 seconds
        (9.25 minutes). The actual Apollo 11 powered descent was
        about 756 seconds, but they throttled down significantly
        during approach and landing.
        """
        if not self.engine_on:
            return 0.0
        return self.thrust_force / self.exhaust_velocity

    @property
    def delta_v_remaining(self) -> float:
        """Remaining velocity-change capability [m/s] via Tsiolkovsky.

        Δv = v_e × ln(m_current / m_dry)
           = Isp × g0 × ln(m_current / m_dry)

        At PDI:        Δv ≈ 2,390 m/s
        Half fuel:     Δv ≈ 1,370 m/s  (NOT half — it's logarithmic!)
        Quarter fuel:  Δv ≈  830 m/s

        The logarithm means Δv is NOT proportional to fuel remaining.
        The first half of the fuel gives ~1,020 m/s of Δv.
        The second half gives ~1,370 m/s.
        The last quarter gives only ~540 m/s.
        """
        if self.propellant <= 0.0:
            return 0.0
        return self.exhaust_velocity * math.log(self.mass / self._dry_mass)

    @property
    def burn_time_remaining(self) -> float:
        """Seconds of propellant remaining at current throttle [s].

        burn_time = propellant / ṁ

        At full thrust: 8,200 / 14.77 ≈ 555 s (9.25 min)
        At 10% throttle: 8,200 / 1.48 ≈ 5,540 s (92 min)

        Returns infinity if engine is off (no consumption).
        """
        flow = self.mass_flow_rate
        if flow <= 0.0:
            return float('inf')
        return self.propellant / flow

    def sync_mass_from_physics(self, new_total_mass: float) -> None:
        """Update propellant after a physics integration step.

        The physics engine integrates total mass as part of the state vector.
        After each step, this method syncs the vehicle model's propellant
        tracking with the physics result.

        If propellant hits zero, sets the fuel_exhausted flag and kills thrust.

        Args:
            new_total_mass: Total mass from the physics integrator [kg].
        """
        new_propellant = new_total_mass - self._dry_mass
        if new_propellant <= 0.0:
            self.propellant = 0.0
            self.fuel_exhausted = True
            self.throttle = 0.0
        else:
            self.propellant = new_propellant

    # ════════════════════════════════════════════════════════════════
    # PHASE 3 — ATTITUDE CONTROL (RCS)
    # ════════════════════════════════════════════════════════════════
    #
    # The Reaction Control System uses small thrusters arranged in four
    # quads around the vehicle. For rotation, opposing quads fire to
    # create a torque couple. The pilot commands rotation direction;
    # the thruster logic fires the appropriate pair.
    #
    # RCS propellant comes from the same pool as the DPS for simplicity
    # in Phase 3. In reality they used separate hypergolic systems.
    # The consumption is tiny (~0.2 kg/s per firing pair) compared to
    # DPS (~14.8 kg/s at full thrust).

    @property
    def rcs_command(self) -> int:
        """Current RCS rotation command: -1 (left), 0 (off), +1 (right)."""
        return self._rcs_command

    def rcs_left(self) -> None:
        """Command counterclockwise (left) rotation.

        In our convention, counterclockwise is negative theta (θ < 0).
        So RCS left applies NEGATIVE torque: τ = -890 N·m.
        """
        self._rcs_command = -1

    def rcs_right(self) -> None:
        """Command clockwise (right) rotation.

        Clockwise is positive theta (θ > 0).
        RCS right applies POSITIVE torque: τ = +890 N·m.
        """
        self._rcs_command = 1

    def rcs_stop(self) -> None:
        """Stop RCS rotation command."""
        self._rcs_command = 0

    @property
    def rcs_torque_value(self) -> float:
        """Current RCS torque [N·m] based on command.

        Returns:
            +890 for right, -890 for left, 0 for off.
        """
        from constants import RCS_ATTITUDE_TORQUE
        return self._rcs_command * RCS_ATTITUDE_TORQUE

