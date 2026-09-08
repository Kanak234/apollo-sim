"""
Apollo Guidance Computer (AGC) simulation — Phase 5.

This module simulates the DSKY (Display and Keyboard) interface that
the Apollo astronauts used to communicate with the guidance computer.

THE DSKY — DISPLAY AND KEYBOARD
────────────────────────────────
The real DSKY had:
    - PROG register: shows current program (P00–P99)
    - VERB register: what to do (V01–V99)
    - NOUN register: what data (N01–N99)
    - Three data registers (R1, R2, R3): 5-digit signed numbers
    - Status lights: PROG, NO ATT, GIMBAL LOCK, etc.
    - Numeric keypad: 0–9, +, -, CLR, PRO, KEY REL, ENTR

The astronaut would key in commands like:
    V06 N62 E (Verb 06 = display, Noun 62 = altitude and rates)
    V16 N68 E (Verb 16 = monitor, Noun 68 = range info)

The computer would continuously update the display registers with
the appropriate data.

PROGRAMS (PROG register):
    P63 = Braking phase
    P64 = Approach phase
    P66 = Rate of Descent
    P67 = Full manual
    P00 = Idle

VERBS:
    V06 = Display (decimal, single-shot)
    V16 = Monitor (decimal, continuous update)
    V37 = Change program

NOUNS (descent-relevant):
    N62 = Velocity, altitude rate, altitude
    N63 = Altitude, altitude rate, downrange
    N64 = Remaining Δv, time to ignition, (reserved)
    N68 = Range to landing site, velocity, altitude rate
    N69 = Vehicle attitude, angular rate, (reserved)

1202 AND 1201 ALARMS
────────────────────
During Apollo 11's descent, the AGC triggered multiple 1202 and 1201
alarms. These are "executive overflow" errors — the computer had too
many tasks queued. In the real mission, Steve Bales in Mission Control
confirmed they were safe to ignore ("GO on that alarm").

We simulate these by randomly triggering alarms under high CPU load
conditions (rapid state changes, many guidance recomputations). The
pilot must acknowledge them to clear the PROG light.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DSKYVerb(Enum):
    """DSKY Verb codes — what to do."""
    V06 = 6    # Display decimal (one-shot)
    V16 = 16   # Monitor decimal (continuous)
    V37 = 37   # Change program
    V50 = 50   # Execute extended verb


class DSKYNoun(Enum):
    """DSKY Noun codes — what data to show."""
    N62 = 62   # Velocity, altitude rate, altitude
    N63 = 63   # Altitude, altitude rate, downrange distance
    N64 = 64   # Remaining Δv, burn time, propellant %
    N68 = 68   # Range to target, total velocity, altitude rate
    N69 = 69   # Vehicle θ (deg), angular rate (deg/s), MOI


@dataclass
class DSKYAlarm:
    """An AGC program alarm.

    Attributes:
        code:       Alarm code (e.g. 1202, 1201).
        message:    Human-readable description.
        time:       MET when the alarm occurred.
        acknowledged: Whether the pilot has dismissed it.
    """
    code: int
    message: str
    time: float
    acknowledged: bool = False


@dataclass
class DSKYRegisters:
    """The three DSKY data registers plus program/verb/noun.

    R1, R2, R3 hold 5-digit signed numbers (as displayed on the DSKY).
    The labels describe what each register currently shows.

    Attributes:
        prog:       Current program number (63, 64, 66, 67, 0).
        verb:       Current verb (6, 16, 37).
        noun:       Current noun (62, 63, 64, 68, 69).
        r1_value:   Register 1 numeric value.
        r1_label:   Register 1 label (e.g. "VEL").
        r2_value:   Register 2 numeric value.
        r2_label:   Register 2 label.
        r3_value:   Register 3 numeric value.
        r3_label:   Register 3 label.
    """
    prog: int = 0
    verb: int = 16
    noun: int = 62

    r1_value: float = 0.0
    r1_label: str = ""
    r2_value: float = 0.0
    r2_label: str = ""
    r3_value: float = 0.0
    r3_label: str = ""

    # Status lights
    prog_light: bool = False    # PROG caution — alarm active
    no_att: bool = False        # NO ATT — attitude reference lost
    gimbal_lock: bool = False   # GIMBAL LOCK (not used until Phase 7+)
    restart: bool = False       # RESTART — computer restarted
    tracker: bool = False       # TRACKER — landing radar lock

    # Input state — for verb/noun entry
    input_mode: str = ""        # "VERB", "NOUN", "PROG", or ""
    input_buffer: str = ""      # Digits being entered


class AGC:
    """Apollo Guidance Computer simulation.

    Manages the DSKY display state, program/verb/noun registers,
    alarm system, and display data routing.

    The AGC is the bridge between the guidance programs (Phase 4)
    and the display. It takes simulation state as input and updates
    the DSKY registers with the appropriate data based on the
    current verb/noun selection.
    """

    def __init__(self) -> None:
        self.registers = DSKYRegisters()
        self.alarms: list[DSKYAlarm] = []
        self._alarm_cooldown: float = 0.0
        self._last_alarm_time: float = 0.0
        self._uplink_activity: bool = False

        # Default display: V16 N62 (monitor altitude/velocity)
        self.registers.verb = 16
        self.registers.noun = 62

    def set_program(self, prog_num: int) -> None:
        """Set the current program number.

        Called when guidance mode changes (P63→P64→P66→P67).

        Args:
            prog_num: Program number (63, 64, 66, 67, or 0 for idle).
        """
        self.registers.prog = prog_num

    def set_verb_noun(self, verb: int, noun: int) -> None:
        """Set the current verb and noun.

        This determines what data appears in R1/R2/R3.

        Args:
            verb: Verb code (6 or 16).
            noun: Noun code (62, 63, 64, 68, or 69).
        """
        self.registers.verb = verb
        self.registers.noun = noun

    def cycle_noun(self) -> None:
        """Cycle through available nouns: 62 → 63 → 64 → 68 → 69 → 62.

        Convenience for the pilot to quickly view different data.
        """
        sequence = [62, 63, 64, 68, 69]
        try:
            idx = sequence.index(self.registers.noun)
            self.registers.noun = sequence[(idx + 1) % len(sequence)]
        except ValueError:
            self.registers.noun = 62

    def trigger_alarm(self, code: int, message: str, time: float) -> None:
        """Trigger a program alarm (e.g., 1202).

        Sets the PROG light and adds to the alarm history.

        Args:
            code:    Alarm code (1201, 1202, etc.).
            message: Description.
            time:    Mission elapsed time.
        """
        alarm = DSKYAlarm(code=code, message=message, time=time)
        self.alarms.append(alarm)
        self.registers.prog_light = True
        self._last_alarm_time = time

    def acknowledge_alarm(self) -> None:
        """Clear the PROG light and acknowledge the latest alarm.

        In the real AGC, pressing KEY REL or PRO would acknowledge alarms.
        We use the PRO key (mapped to Enter or a specific key).
        """
        for alarm in reversed(self.alarms):
            if not alarm.acknowledged:
                alarm.acknowledged = True
                break
        # Clear PROG light if all alarms acknowledged
        if all(a.acknowledged for a in self.alarms):
            self.registers.prog_light = False

    def maybe_trigger_1202(self, time: float, thrust_on: bool,
                           guidance_active: bool) -> None:
        """Randomly trigger a 1202 alarm under high-load conditions.

        EXECUTIVE OVERFLOW — 1202
        ─────────────────────────
        The real 1202 occurred because the rendezvous radar was
        flooding the computer with interrupts during descent. The
        AGC had too many tasks queued and the executive (scheduler)
        overflowed.

        We simulate this probabilistically: when the engine is firing
        AND guidance is doing recomputations, there's a small chance
        of a 1202. This creates the authentic tension without being
        unfairly frequent.

        Historical note: Apollo 11 had five alarms during descent
        (three 1202s and two 1201s). The decision to continue was
        one of the most critical calls in spaceflight history.

        Args:
            time:            Current MET [s].
            thrust_on:       Whether engine is firing.
            guidance_active: Whether guidance is recomputing.
        """
        # Cooldown between alarms (at least 30 seconds)
        if time - self._last_alarm_time < 30.0:
            return

        # Only trigger when computer is busy (engine + guidance)
        if not (thrust_on and guidance_active):
            return

        # ~0.2% chance per second (checked at ~10 Hz) ≈ 1 alarm per ~8 minutes
        # This gives roughly 1 alarm per descent — historically accurate
        if random.random() < 0.002:
            code = random.choice([1202, 1202, 1201])  # 1202 more common
            if code == 1202:
                msg = "EXECUTIVE OVERFLOW — cycle steal"
            else:
                msg = "EXECUTIVE OVERFLOW — no VAC areas"
            self.trigger_alarm(code, msg, time)
            print(f"\n  ⚠ PROGRAM ALARM {code} — {msg}")
            print(f"    MET {time:.0f}s — Press ENTER to acknowledge")
            print()

    def update_display(
        self,
        altitude: float,
        velocity: float,
        vx: float,
        vy: float,
        theta: float,
        omega: float,
        mass: float,
        delta_v: float,
        burn_time: float,
        propellant: float,
        propellant_max: float,
        x: float,
        landing_site_x: float,
        time_elapsed: float,
        moi: float = 0.0,
    ) -> None:
        """Update R1/R2/R3 based on current verb/noun.

        This is the core DSKY data routing: given the verb/noun selection,
        fill the three registers with the appropriate data.

        NOUN DEFINITIONS:
            N62: R1=velocity(m/s)      R2=alt_rate(m/s)     R3=altitude(m)
            N63: R1=altitude(m)        R2=alt_rate(m/s)     R3=downrange(m)
            N64: R1=Δv_remaining(m/s)  R2=burn_time(s)      R3=propellant(%)
            N68: R1=range_to_pad(m)    R2=total_vel(m/s)    R3=alt_rate(m/s)
            N69: R1=theta(deg)         R2=omega(deg/s)      R3=MOI(kg·m²)
        """
        noun = self.registers.noun
        v_total = math.sqrt(vx * vx + vy * vy)
        range_to_pad = abs(x - landing_site_x)
        prop_pct = (propellant / propellant_max * 100.0) if propellant_max > 0 else 0

        if noun == 62:
            self.registers.r1_label = "VEL   m/s"
            self.registers.r1_value = v_total
            self.registers.r2_label = "VDOT  m/s"
            self.registers.r2_value = vy
            self.registers.r3_label = "ALT     m"
            self.registers.r3_value = altitude

        elif noun == 63:
            self.registers.r1_label = "ALT     m"
            self.registers.r1_value = altitude
            self.registers.r2_label = "VDOT  m/s"
            self.registers.r2_value = vy
            self.registers.r3_label = "DWNRG   m"
            self.registers.r3_value = range_to_pad

        elif noun == 64:
            self.registers.r1_label = "DV    m/s"
            self.registers.r1_value = delta_v
            self.registers.r2_label = "BURN    s"
            self.registers.r2_value = burn_time if burn_time < 9999 else 9999
            self.registers.r3_label = "PROP    %"
            self.registers.r3_value = prop_pct

        elif noun == 68:
            self.registers.r1_label = "RNG     m"
            self.registers.r1_value = range_to_pad
            self.registers.r2_label = "VEL   m/s"
            self.registers.r2_value = v_total
            self.registers.r3_label = "VDOT  m/s"
            self.registers.r3_value = vy

        elif noun == 69:
            self.registers.r1_label = "θ     deg"
            self.registers.r1_value = math.degrees(theta)
            self.registers.r2_label = "ω    °/s"
            self.registers.r2_value = math.degrees(omega)
            self.registers.r3_label = "MOI kg·m²"
            self.registers.r3_value = moi

        else:
            # Unknown noun — blank registers
            self.registers.r1_label = "---"
            self.registers.r1_value = 0.0
            self.registers.r2_label = "---"
            self.registers.r2_value = 0.0
            self.registers.r3_label = "---"
            self.registers.r3_value = 0.0

        # Landing radar tracker light — locks on below 12 km
        self.registers.tracker = altitude < 12_000.0

    def get_active_alarm(self) -> Optional[DSKYAlarm]:
        """Get the most recent unacknowledged alarm, if any."""
        for alarm in reversed(self.alarms):
            if not alarm.acknowledged:
                return alarm
        return None

    def format_register(self, value: float, width: int = 6) -> str:
        """Format a number for DSKY display.

        The real DSKY displayed 5-digit numbers with sign (+/-).
        We use a similar format: sign + digits.

        Rules:
            - Magnitude < 1000: show 1 decimal place
            - Magnitude < 100000: show integer
            - Otherwise: scientific notation

        Args:
            value: Numeric value.
            width: Display width.

        Returns:
            Formatted string.
        """
        if abs(value) < 1000:
            return f"{value:+{width}.1f}"
        elif abs(value) < 100000:
            return f"{value:+{width}.0f}"
        else:
            return f"{value:+.1e}"
