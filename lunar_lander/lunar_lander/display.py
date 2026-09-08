"""
Display renderer for the Lunar Module descent simulator.

Handles all pygame rendering: terrain, vehicle, engine effects, star field,
and the DSKY-style HUD overlay. The rendering is fully decoupled from the
physics timestep — this module only READS state, never modifies it.

Visual design is inspired by the Apollo Guidance Computer's DSKY unit:
green electroluminescent digits on a black background, fixed-width font,
numeric register readouts. Phase 5 will implement the full DSKY with
verb/noun entry; Phase 1 shows a simplified but authentic-feeling HUD.

Display architecture:
─────────────────────
    ┌─────────────────────────────┬──────────────┐
    │                             │  ╔═ DSKY ═╗  │
    │     Simulation viewport     │  PROG  01    │
    │                             │  VERB  NOUN  │
    │         ★         ★        │   06    62   │
    │    ★                        │  ──────────  │
    │              △              │  R1 VEL      │
    │              ▽ (flame)      │   -50.0      │
    │                             │  R2 ALT RATE │
    │                             │   -50.0      │
    │  ═══════════════════════════│  R3 ALT      │
    │  ░░░░ lunar surface ░░░░░░░│   2000.0     │
    └─────────────────────────────┴──────────────┘
"""

from __future__ import annotations

import pygame
import random
import math

from constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    RENDER_FPS,
    PHASE1_START_ALT,
    MAX_VERTICAL_SPEED,
)


# ============================================================================
# COLOR PALETTE
# ============================================================================
# Colors chosen to evoke the Apollo-era aesthetic: green phosphor displays,
# the grey lunar surface, and the blackness of space.

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# DSKY display colors — the real DSKY used green electroluminescent panels
DSKY_GREEN = (0, 255, 128)          # Bright readout digits
DSKY_DIM_GREEN = (0, 128, 64)       # Labels and inactive elements
DSKY_BG = (10, 12, 10)              # Near-black panel background
DSKY_BORDER = (0, 180, 90)          # Panel border

# Lunar surface
LUNAR_GREY = (90, 90, 85)           # Main regolith color
LUNAR_GREY_DARK = (60, 60, 55)      # Shadows and craters
LUNAR_GREY_LIGHT = (110, 108, 100)  # Highlights

# Engine exhaust — layered for depth
FLAME_CORE = (255, 255, 220)        # White-hot core
FLAME_YELLOW = (255, 220, 80)       # Inner flame
FLAME_ORANGE = (255, 160, 40)       # Mid flame
FLAME_RED = (255, 80, 20)           # Outer tips

# Status indicators
LANDED_GREEN = (0, 255, 100)
CRASHED_RED = (255, 40, 40)
WARNING_AMBER = (255, 180, 0)

# ============================================================================
# LAYOUT CONSTANTS
# ============================================================================

TERRAIN_HEIGHT = 60        # px reserved for terrain at bottom
HUD_WIDTH = 330            # px width of the right-side HUD panel
HUD_X = WINDOW_WIDTH - HUD_WIDTH  # HUD panel left edge
VIEW_WIDTH = WINDOW_WIDTH - HUD_WIDTH  # simulation viewport width
HUD_PADDING = 15           # internal padding within HUD


class Display:
    """Manages the pygame window and all visual rendering.

    The Display class is a pure renderer — it takes simulation state as
    arguments and draws it. It never modifies simulation state. This
    separation is critical: it means the physics is testable without
    any display, and the display can be swapped out without affecting
    the simulation.
    """

    def __init__(self) -> None:
        """Initialize pygame, create the window, and set up fonts."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Apollo LM Descent Simulator — Phase 6")
        self.clock = pygame.time.Clock()

        # ── Fonts ──
        # Use monospaced fonts for DSKY authenticity.
        # pygame.font.SysFont searches system fonts; "couriernew" or "courier"
        # should be available on all platforms. Fallback to pygame's default.
        mono_name = "couriernew"
        self.font_dsky_large = pygame.font.SysFont(mono_name, 28, bold=True)
        self.font_dsky_medium = pygame.font.SysFont(mono_name, 22, bold=True)
        self.font_dsky_small = pygame.font.SysFont(mono_name, 16)
        self.font_dsky_title = pygame.font.SysFont(mono_name, 14, bold=True)
        self.font_verdict = pygame.font.SysFont(mono_name, 42, bold=True)
        self.font_info = pygame.font.SysFont(mono_name, 18)
        self.font_debug = pygame.font.SysFont(mono_name, 14)

        # ── Pre-generate star field ──
        # Random positions and varying brightness simulate the star field
        # visible from the lunar surface (no atmosphere to block them).
        random.seed(42)  # Fixed seed for consistent stars between runs
        self.stars: list[tuple[int, int, int]] = [
            (
                random.randint(5, VIEW_WIDTH - 5),
                random.randint(5, WINDOW_HEIGHT - TERRAIN_HEIGHT - 10),
                random.randint(140, 255),
            )
            for _ in range(150)
        ]

        # ── HUD surface with alpha channel for semi-transparency ──
        self.hud_surface = pygame.Surface(
            (HUD_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA
        )

        # ── Altitude scale tracking ──
        # The view auto-scales to keep the LM visible
        self._max_display_alt = PHASE1_START_ALT * 1.15

    def _alt_to_screen_y(self, altitude: float) -> int:
        """Convert altitude in meters to screen Y pixel coordinate.

        Coordinate systems:
            Simulation: altitude = 0 at surface, positive UP
            Screen:     y = 0 at top of window, positive DOWN

        So the mapping inverts the vertical axis:
            screen_y = terrain_top - (altitude / max_alt) × available_height

        Args:
            altitude: Height above surface [m].

        Returns:
            Screen Y coordinate [px].
        """
        terrain_top = WINDOW_HEIGHT - TERRAIN_HEIGHT
        view_height = terrain_top - 40  # top margin
        frac = max(0.0, min(1.0, altitude / self._max_display_alt))
        return int(terrain_top - frac * view_height)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def render(
        self,
        altitude: float,
        velocity: float,
        time_elapsed: float,
        thrust_on: bool,
        status: str,
        paused: bool = False,
        debug: bool = False,
        throttle: float = 0.0,
        propellant: float = 8200.0,
        propellant_max: float = 8200.0,
        mass: float = 15100.0,
        delta_v: float = 0.0,
        burn_time: float = float('inf'),
        fuel_warning: bool = False,
        x: float = 0.0,
        vx: float = 0.0,
        theta: float = 0.0,
        omega: float = 0.0,
        rcs_cmd: int = 0,
        landing_site_x: float = 0.0,
        violations: list = None,
        moi: float = 0.0,
        agc_registers=None,
        guidance_mode: str = '',
        rod_target: float = 0.0,
        # Phase 6 additions
        terrain=None,
        radar_locked: bool = False,
        radar_alt: float = 0.0,
    ) -> None:
        """Render one complete frame of the simulation.

        Called once per display frame (60 FPS). Draws everything in order
        from back to front (painter's algorithm):
            1. Black sky background
            2. Stars
            3. Lunar terrain
            4. Vehicle with optional engine flame
            5. HUD panel (right side)
            6. Verdict overlay (if landed/crashed)
            7. Pause overlay (if paused)
            8. Debug overlay (if toggled)
        """
        # Clear to black (space)
        self.screen.fill(BLACK)

        camera_x = x

        # 1. Stars
        self._draw_stars()

        # 2. Lunar terrain
        self._draw_terrain(camera_x, landing_site_x, terrain)

        # 3. Vehicle
        lm_x = VIEW_WIDTH // 2
        lm_y = self._alt_to_screen_y(altitude)
        self._draw_lander(lm_x, lm_y, thrust_on, status, throttle, theta)

        # 4. Altitude scale (left side reference lines)
        self._draw_altitude_scale()

        # 5. HUD panel
        if agc_registers is not None:
            self._draw_dsky(
                agc_registers, guidance_mode, rod_target,
                altitude, velocity, time_elapsed, thrust_on, status,
                throttle, propellant, propellant_max, delta_v, burn_time, fuel_warning,
                vx, theta, x, landing_site_x, rcs_cmd, radar_locked, radar_alt
            )
        else:
            self._draw_hud(
                altitude, velocity, time_elapsed, thrust_on, status,
                throttle, propellant, propellant_max, delta_v, burn_time, fuel_warning,
                vx, theta, x, landing_site_x, rcs_cmd, radar_locked, radar_alt
            )

        # 6. Verdict overlay
        if status in ("landed", "crashed"):
            self._draw_verdict(status, velocity, time_elapsed, propellant, delta_v, violations)

        # 7. Pause overlay
        if paused and status == "flying":
            self._draw_pause_overlay()

        # 8. Debug info
        if debug:
            self._draw_debug(altitude, velocity, thrust_on, throttle, propellant, mass, x, vx, theta, omega, moi, agc_registers, guidance_mode, terrain, radar_locked, radar_alt)

        pygame.display.flip()
        self.clock.tick(RENDER_FPS)

    def cleanup(self) -> None:
        """Clean up pygame resources. Call before exit."""
        pygame.quit()

    # ========================================================================
    # PRIVATE RENDERING METHODS
    # ========================================================================

    def _draw_stars(self) -> None:
        """Draw the pre-generated star field."""
        for x, y, brightness in self.stars:
            # Slight blue tint to some stars for realism
            b_component = min(255, brightness + 15)
            self.screen.set_at((x, y), (brightness, brightness, b_component))
            # Brighter stars get a second pixel for visibility
            if brightness > 220:
                self.screen.set_at((x + 1, y), (brightness - 40, brightness - 40, brightness - 30))

    def _draw_terrain(self, camera_x: float = 0.0, landing_site_x: float = 0.0, terrain=None) -> None:
        """Draw the flat lunar surface with visual detail."""
        terrain_top = WINDOW_HEIGHT - TERRAIN_HEIGHT
        offset_x = (VIEW_WIDTH // 2) - camera_x

        # Main regolith surface and subsurface layer
        pygame.draw.rect(
            self.screen, LUNAR_GREY,
            (0, terrain_top, VIEW_WIDTH, TERRAIN_HEIGHT),
        )
        pygame.draw.rect(
            self.screen, LUNAR_GREY_DARK,
            (0, terrain_top + 40, VIEW_WIDTH, TERRAIN_HEIGHT - 40),
        )

        terrain_points = []
        start_world_x = camera_x - VIEW_WIDTH // 2

        if terrain is not None:
            pad_height = terrain.height_at(landing_site_x)
            profile = terrain.get_terrain_profile(camera_x, VIEW_WIDTH)
            
            if isinstance(profile, tuple) and len(profile) == 2:
                xs, hs = profile
                points_iter = zip(xs, hs)
            else:
                points_iter = profile
                
            for world_x, world_h in points_iter:
                px = world_x - start_world_x
                screen_h = (world_h - pad_height) / 2.0
                screen_y = terrain_top - screen_h
                terrain_points.append((px, screen_y))
        else:
            for px in range(0, VIEW_WIDTH + 1, 10):
                world_x = start_world_x + px
                if abs(world_x - landing_site_x) < 50:
                    h = 0
                else:
                    dist = abs(world_x - landing_site_x) - 50
                    h = 5 * math.sin(world_x / 20.0) + 10 * math.sin(world_x / 100.0)
                    damp = min(1.0, dist / 50.0)
                    h *= damp
                screen_y = terrain_top - h
                terrain_points.append((px, screen_y))

        if terrain_points:
            poly_points = [(0, WINDOW_HEIGHT), (VIEW_WIDTH, WINDOW_HEIGHT)] + list(reversed(terrain_points))
            pygame.draw.polygon(self.screen, LUNAR_GREY, poly_points)

            # Surface line
            if len(terrain_points) > 1:
                pygame.draw.lines(self.screen, LUNAR_GREY_LIGHT, False, terrain_points, 2)

        # Craters — ellipses for perspective
        crater_data = [
            (80, 18, 8),     # (center_x, radius_x, radius_y)
            (220, 30, 12),
            (380, 12, 5),
            (500, 22, 9),
            (150, 8, 4),
            (450, 15, 6),
            (600, 10, 4),
        ]
        # Spread craters out a bit over world coordinates
        crater_world_positions = []
        for i, (cx, rx, ry) in enumerate(crater_data):
            crater_world_positions.append((cx * 2 - 600, rx, ry))
            crater_world_positions.append((cx * 3 + 400, rx, ry))
            crater_world_positions.append((-cx * 2, rx, ry))

        for cx_world, rx, ry in crater_world_positions:
            cx = offset_x + cx_world
            if -rx <= cx <= VIEW_WIDTH + rx:
                if terrain is not None:
                    pad_height = terrain.height_at(landing_site_x)
                    c_world_y = terrain.height_at(cx_world)
                    c_screen_y = terrain_top - (c_world_y - pad_height) / 2.0
                else:
                    c_screen_y = terrain_top
                
                # Draw crater
                pygame.draw.ellipse(
                    self.screen, LUNAR_GREY_DARK,
                    (cx - rx, c_screen_y + 3, rx * 2, ry * 2),
                )
                pygame.draw.arc(
                    self.screen, LUNAR_GREY_LIGHT,
                    (cx - rx, c_screen_y + 3, rx * 2, ry * 2),
                    math.pi * 0.8, math.pi * 1.8, 1,
                )

        # Landing zone marker (flat pad with H)
        pad_screen_x = offset_x + landing_site_x
        if -50 <= pad_screen_x <= VIEW_WIDTH + 50:
            if terrain is not None:
                pad_screen_y = terrain_top
            else:
                pad_screen_y = terrain_top
                
            pad_rect = pygame.Rect(pad_screen_x - 50, pad_screen_y, 100, 4)
            pygame.draw.rect(self.screen, (150, 150, 150), pad_rect)

            h_color = (200, 200, 0)
            h_x = pad_screen_x
            h_y = pad_screen_y - 10
            pygame.draw.line(self.screen, h_color, (h_x - 10, h_y), (h_x - 10, h_y + 10), 2)
            pygame.draw.line(self.screen, h_color, (h_x + 10, h_y), (h_x + 10, h_y + 10), 2)
            pygame.draw.line(self.screen, h_color, (h_x - 10, h_y + 5), (h_x + 10, h_y + 5), 2)

    def _draw_lander(
        self, x: int, y: int, thrust_on: bool, status: str, throttle: float, theta: float
    ) -> None:
        """Draw the LM as a simplified 2D glyph onto a rotated surface."""
        if status == "crashed":
            # Draw wreckage directly (no rotation needed for scattered debris)
            self._draw_wreckage(x, y)
            return

        # Create temporary surface
        surf_size = 120
        lm_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = surf_size // 2

        # ── Descent stage (lower box) ──
        body_w, body_h = 32, 20
        descent_rect = pygame.Rect(center - body_w // 2, center - body_h, body_w, body_h)
        pygame.draw.rect(lm_surf, (190, 190, 185), descent_rect)
        pygame.draw.rect(lm_surf, (140, 140, 135), descent_rect, 1)

        for dy in range(3, body_h - 2, 4):
            pygame.draw.line(
                lm_surf, (200, 180, 80),
                (center - body_w // 2 + 2, center - body_h + dy),
                (center + body_w // 2 - 2, center - body_h + dy), 1,
            )

        # ── Ascent stage (upper cabin) ──
        asc_pts = [
            (center - 14, center - body_h),
            (center + 14, center - body_h),
            (center + 10, center - body_h - 20),
            (center - 10, center - body_h - 20),
        ]
        pygame.draw.polygon(lm_surf, (175, 175, 170), asc_pts)
        pygame.draw.polygon(lm_surf, (130, 130, 125), asc_pts, 1)

        window_pts = [
            (center - 5, center - body_h - 4),
            (center + 5, center - body_h - 4),
            (center, center - body_h - 14),
        ]
        pygame.draw.polygon(lm_surf, (80, 160, 220), window_pts)

        # ── S-band antenna (top) ──
        pygame.draw.line(
            lm_surf, (160, 160, 155),
            (center, center - body_h - 20), (center, center - body_h - 28), 1,
        )
        pygame.draw.circle(
            lm_surf, (160, 160, 155),
            (center, center - body_h - 28), 3, 1,
        )

        # ── Landing legs ──
        leg_base_y = center + 12
        pygame.draw.line(
            lm_surf, (160, 160, 155),
            (center - body_w // 2, center), (center - body_w // 2 - 14, leg_base_y), 2,
        )
        pygame.draw.line(
            lm_surf, (160, 160, 155),
            (center + body_w // 2, center), (center + body_w // 2 + 14, leg_base_y), 2,
        )
        pygame.draw.circle(
            lm_surf, (140, 140, 135),
            (center - body_w // 2 - 14, leg_base_y), 3,
        )
        pygame.draw.circle(
            lm_surf, (140, 140, 135),
            (center + body_w // 2 + 14, leg_base_y), 3,
        )

        # ── Engine nozzle ──
        nozzle_pts = [
            (center - 7, center),
            (center + 7, center),
            (center + 11, center + 10),
            (center - 11, center + 10),
        ]
        pygame.draw.polygon(lm_surf, (150, 150, 145), nozzle_pts)
        pygame.draw.polygon(lm_surf, (120, 120, 115), nozzle_pts, 1)

        # ── Engine flame ──
        if thrust_on and throttle > 0:
            self._draw_flame(lm_surf, center, center + 10, throttle)

        # Rotate the surface
        # Pygame rotates counterclockwise; theta is clockwise (standard graphics rotation).
        # Wait, if theta is clockwise, we pass -math.degrees(theta) to rotate it clockwise?
        # pygame.transform.rotate(..., angle) rotates counterclockwise for positive angle.
        # So angle = -math.degrees(theta).
        rotated_surf = pygame.transform.rotate(lm_surf, -math.degrees(theta))
        rot_rect = rotated_surf.get_rect(center=(x, y))
        self.screen.blit(rotated_surf, rot_rect)

    def _draw_flame(self, surf: pygame.Surface, x: int, y: int, throttle: float) -> None:
        """Draw a multi-layered engine exhaust plume on a surface."""
        base_length = int(10 + 45 * throttle)
        flicker_amp = int(12 * throttle)
        flicker = random.randint(-int(flicker_amp * 0.7), flicker_amp)
        flame_len = max(5, base_length + flicker)
        
        width_scale = 0.5 + 0.5 * throttle

        outer_w = int(18 * width_scale)
        outer_pts = [
            (x - outer_w // 2 + random.randint(-2, 2), y),
            (x + outer_w // 2 + random.randint(-2, 2), y),
            (x + random.randint(-4, 4), y + flame_len),
        ]
        pygame.draw.polygon(surf, FLAME_RED, outer_pts)

        mid_w = int(12 * width_scale)
        mid_pts = [
            (x - mid_w // 2, y),
            (x + mid_w // 2, y),
            (x + random.randint(-2, 2), y + int(flame_len * 0.85)),
        ]
        pygame.draw.polygon(surf, FLAME_ORANGE, mid_pts)

        inner_w = int(8 * width_scale)
        inner_pts = [
            (x - inner_w // 2, y),
            (x + inner_w // 2, y),
            (x + random.randint(-1, 1), y + int(flame_len * 0.65)),
        ]
        pygame.draw.polygon(surf, FLAME_YELLOW, inner_pts)

        core_pts = [
            (x - 2, y),
            (x + 2, y),
            (x, y + int(flame_len * 0.4)),
        ]
        pygame.draw.polygon(surf, FLAME_CORE, core_pts)

        for _ in range(int(4 * throttle) + 1):
            px = x + random.randint(-15, 15)
            py = y + random.randint(int(flame_len * 0.5), flame_len + 10)
            size = random.randint(1, 2)
            pygame.draw.circle(
                surf,
                (255, random.randint(100, 200), 20),
                (px, py), size,
            )

    def _draw_wreckage(self, x: int, y: int) -> None:
        """Draw crash debris instead of an intact LM."""
        random.seed(12345)
        for _ in range(15):
            fx = x + random.randint(-30, 30)
            fy = y + random.randint(-15, 5)
            fw = random.randint(3, 12)
            fh = random.randint(2, 8)
            color = random.choice([
                (150, 150, 145),
                (200, 180, 80),
                (100, 100, 95),
            ])
            pygame.draw.rect(self.screen, color, (fx, fy, fw, fh))

        for _ in range(8):
            sx = x + random.randint(-25, 25)
            sy = y + random.randint(-20, 0)
            sr = random.randint(5, 15)
            pygame.draw.circle(self.screen, (80, 80, 75), (sx, sy), sr)
        random.seed()

    def _draw_altitude_scale(self) -> None:
        """Draw altitude reference lines on the left side of the viewport."""
        terrain_top = WINDOW_HEIGHT - TERRAIN_HEIGHT

        interval = 500.0
        alt = interval
        while alt < self._max_display_alt:
            y = self._alt_to_screen_y(alt)
            if y > 30:
                pygame.draw.line(
                    self.screen, (40, 60, 40),
                    (0, y), (VIEW_WIDTH, y), 1,
                )
                label = self.font_debug.render(f"{alt:.0f}m", True, (50, 80, 50))
                self.screen.blit(label, (5, y - 12))
            alt += interval

    def _draw_dsky(
        self,
        agc_registers,
        guidance_mode: str,
        rod_target: float,
        altitude: float,
        velocity: float,
        time_elapsed: float,
        thrust_on: bool,
        status: str,
        throttle: float,
        propellant: float,
        propellant_max: float,
        delta_v: float,
        burn_time: float,
        fuel_warning: bool,
        vx: float,
        theta: float,
        x: float,
        landing_site_x: float,
        rcs_cmd: int,
        radar_locked: bool = False,
        radar_alt: float = 0.0,
    ) -> None:
        """Draw the authentic Phase 5 DSKY interface."""
        self.hud_surface.fill(DSKY_BG)
        y = HUD_PADDING

        self._hud_text("╔══════════ DSKY ══════════╗", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 24

        self._hud_text("  PROG    VERB    NOUN", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        
        # PROG label logic
        prog_color = DSKY_GREEN
        if agc_registers.prog_light:
            if (pygame.time.get_ticks() % 1000) < 500:
                prog_color = WARNING_AMBER
            else:
                prog_color = DSKY_BG  # "Flash" by drawing in bg color (invisible)

        self._hud_text(f"{agc_registers.prog:02d}", y, self.font_dsky_large, prog_color, x_offset=HUD_PADDING + 20)
        self._hud_text(f"{agc_registers.verb:02d}", y, self.font_dsky_large, DSKY_GREEN, x_offset=HUD_PADDING + 100)
        self._hud_text(f"{agc_registers.noun:02d}", y, self.font_dsky_large, DSKY_GREEN, x_offset=HUD_PADDING + 180)
        
        y += 32

        self._hud_separator(y)
        y += 12

        # Registers
        # R1: Vel
        self._hud_text("  R1  VEL   m/s", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        self._hud_text(f"{vx:+08.1f}", y, self.font_dsky_large, DSKY_GREEN, x_offset=HUD_PADDING + 20)
        y += 28

        # R2: VDOT (Alt Rate)
        self._hud_text("  R2  VDOT  m/s", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        self._hud_text(f"{velocity:+08.1f}", y, self.font_dsky_large, DSKY_GREEN, x_offset=HUD_PADDING + 20)
        y += 28

        # R3: ALT
        self._hud_text("  R3  ALT     m", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        self._hud_text(f"{altitude:+08.1f}", y, self.font_dsky_large, DSKY_GREEN, x_offset=HUD_PADDING + 20)
        y += 28

        self._hud_separator(y)
        y += 12

        # Status lights
        # [PROG] [NO ATT] [TRK]
        light_y = y
        self._draw_status_light("PROG", HUD_PADDING + 10, light_y, WARNING_AMBER if agc_registers.prog_light else DSKY_DIM_GREEN)
        self._draw_status_light("NO ATT", HUD_PADDING + 90, light_y, CRASHED_RED if False else DSKY_DIM_GREEN) # No specific NO ATT logic yet
        self._draw_status_light("TRK", HUD_PADDING + 190, light_y, DSKY_GREEN if status == 'flying' else DSKY_DIM_GREEN)
        y += 24

        light_y2 = y
        if radar_locked:
            self._draw_status_light("RDR LOCK", HUD_PADDING + 10, light_y2, DSKY_GREEN)
            rdr_alt_str = f"{radar_alt:07.1f}"
        else:
            rdr_color = DSKY_DIM_GREEN if altitude > 15240 else WARNING_AMBER
            self._draw_status_light("RDR ---", HUD_PADDING + 10, light_y2, rdr_color)
            rdr_alt_str = "  -----"
            
        self._hud_text(f"ALT(R) {rdr_alt_str}m", light_y2, self.font_dsky_title, DSKY_GREEN if radar_locked else DSKY_DIM_GREEN, x_offset=HUD_PADDING + 90)
        y += 30

        self._hud_separator(y)
        y += 12

        # Throttle and Propellant bars
        throttle_pct = int(throttle * 100)
        self._hud_text(f"THR {throttle_pct:3d}%", y, self.font_dsky_title, DSKY_DIM_GREEN)
        
        bar_w = 160
        bar_h = 10
        bar_x = HUD_PADDING + 90
        bar_y = y
        
        pygame.draw.rect(self.hud_surface, DSKY_DIM_GREEN, (bar_x, bar_y, bar_w, bar_h), 1)
        fill_w = int(bar_w * throttle)
        if fill_w > 0:
            pygame.draw.rect(self.hud_surface, DSKY_GREEN, (bar_x, bar_y + 1, fill_w, bar_h - 2))
        y += 20
        
        prop_pct = (propellant / propellant_max * 100) if propellant_max > 0 else 0
        self._hud_text(f"PROP {int(prop_pct):2d}%", y, self.font_dsky_title, DSKY_DIM_GREEN)
        bar_y = y
        pygame.draw.rect(self.hud_surface, DSKY_DIM_GREEN, (bar_x, bar_y, bar_w, bar_h), 1)
        pfill_w = int(bar_w * (propellant / propellant_max)) if propellant_max > 0 else 0
        p_color = DSKY_GREEN
        if prop_pct <= 10: p_color = CRASHED_RED
        elif prop_pct <= 20: p_color = WARNING_AMBER
        if pfill_w > 0:
            pygame.draw.rect(self.hud_surface, p_color, (bar_x, bar_y + 1, pfill_w, bar_h - 2))
        y += 20

        # Delta V and Burn Time
        if math.isinf(burn_time):
            bt_str = "---s"
        else:
            bt_str = f"{int(burn_time)}s"
        self._hud_text(f"Δv {int(delta_v)}  BURN {bt_str}", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 20

        self._hud_separator(y)
        y += 12

        # Guidance Mode & MET
        g_mode_str = f"{guidance_mode}"
        if guidance_mode == "P66":
            g_mode_str += f" ROD {rod_target:+.1f} m/s"
        self._hud_text(g_mode_str, y, self.font_dsky_medium, DSKY_GREEN)
        y += 24

        total_secs = time_elapsed
        hrs = int(total_secs // 3600)
        mins = int((total_secs % 3600) // 60)
        secs = int(total_secs % 60)
        self._hud_text(f"MET {hrs:02d}:{mins:02d}:{secs:02d}", y, self.font_dsky_medium, DSKY_GREEN)
        y += 28

        self._hud_separator(y)
        y += 12

        # Fuel Warn
        if fuel_warning and (pygame.time.get_ticks() % 1000) < 500:
            self._hud_text("FUEL WARN", y, self.font_dsky_large, CRASHED_RED)
        y += 32

        self._hud_separator(y)
        y += 12
        
        self._hud_text("── CONTROLS ──", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        controls = [
            ("N", "Cycle DSKY Noun"),
            ("ENTER", "Acknowledge Alarm"),
            ("1/2/3/4", "P63/P64/P66/P67"),
            ("A/D", "RCS Left/Right"),
            ("W/S", "Throttle/ROD"),
            ("SPACE", "Toggle Engine"),
            ("P", "Pause"),
            ("R", "Reset"),
            ("F1", "Debug"),
            ("ESC", "Quit"),
        ]
        for key, action in controls:
            self._hud_text(f"{key:<10s}{action}", y, self.font_dsky_small, DSKY_DIM_GREEN)
            y += 14

        self._hud_text("╚══════════════════════════╝", WINDOW_HEIGHT - 20, self.font_dsky_title, DSKY_DIM_GREEN)

        pygame.draw.rect(
            self.hud_surface, DSKY_BORDER,
            (0, 0, HUD_WIDTH, WINDOW_HEIGHT), 2,
        )
        self.screen.blit(self.hud_surface, (HUD_X, 0))

    def _draw_status_light(self, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Helper to draw a small lit rectangle with text."""
        rect = pygame.Rect(x, y, 75, 20)
        pygame.draw.rect(self.hud_surface, color, rect, 2 if color == DSKY_DIM_GREEN else 0)
        text_color = DSKY_DIM_GREEN if color == DSKY_DIM_GREEN else BLACK
        surf = self.font_dsky_title.render(text, True, text_color)
        t_rect = surf.get_rect(center=rect.center)
        self.hud_surface.blit(surf, t_rect)

    def _draw_hud(
        self,
        altitude: float,
        velocity: float,
        time_elapsed: float,
        thrust_on: bool,
        status: str,
        throttle: float,
        propellant: float,
        propellant_max: float,
        delta_v: float,
        burn_time: float,
        fuel_warning: bool,
        vx: float,
        theta: float,
        x: float,
        landing_site_x: float,
        rcs_cmd: int,
        radar_locked: bool = False,
        radar_alt: float = 0.0,
    ) -> None:
        """Draw the DSKY-style HUD panel on the right side."""
        self.hud_surface.fill((5, 8, 5, 220))
        y = HUD_PADDING

        self._hud_text("══ APOLLO  DSKY ══", y, self.font_dsky_title, DSKY_GREEN)
        y += 20

        self._hud_text("PROG", y, self.font_dsky_title, DSKY_DIM_GREEN)
        self._hud_text("01", y, self.font_dsky_medium, DSKY_GREEN, x_offset=90)
        y += 24

        self._hud_separator(y)
        y += 8

        self._hud_text("VERB      NOUN", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        self._hud_text("06", y, self.font_dsky_medium, DSKY_GREEN, x_offset=HUD_PADDING + 5)
        self._hud_text("62", y, self.font_dsky_medium, DSKY_GREEN, x_offset=145)
        y += 28

        self._hud_separator(y)
        y += 8

        # Reduce spacing a bit to fit new stats
        y_inc_small = 14
        y_inc_large = 24

        self._hud_text("R1  VERT VEL  m/s", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        vel_str = f"{velocity:+08.1f}"
        self._hud_text(vel_str, y, self.font_dsky_large, DSKY_GREEN)
        y += y_inc_large

        self._hud_text("R2  ALTITUDE    m", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        alt_str = f"{max(0.0, altitude):08.1f}"
        self._hud_text(alt_str, y, self.font_dsky_large, DSKY_GREEN)
        y += y_inc_large - 4
        
        self._hud_text("    RADAR ALT   m", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        if radar_locked:
            ralt_str = f"{max(0.0, radar_alt):08.1f}"
            self._hud_text(ralt_str, y, self.font_dsky_large, DSKY_GREEN)
        else:
            self._hud_text("   -----", y, self.font_dsky_large, WARNING_AMBER if altitude <= 15240 else DSKY_DIM_GREEN)
        y += y_inc_large - 4

        self._hud_separator(y)
        y += 8

        # ── HORIZ VEL ──
        self._hud_text("HORIZ VEL", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        hvel_color = DSKY_GREEN
        if altitude < 10.0 and abs(vx) > 1.0 and status == "flying":
            hvel_color = WARNING_AMBER
        self._hud_text(f"{vx:+08.1f} m/s", y, self.font_dsky_medium, hvel_color)
        y += y_inc_large

        # ── VEHICLE θ ──
        self._hud_text("VEHICLE θ", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        self._hud_text(f"{math.degrees(theta):+08.1f} °", y, self.font_dsky_medium, DSKY_GREEN)
        y += y_inc_large

        # ── DOWNRANGE ──
        self._hud_text("DOWNRANGE", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        downrange = abs(x - landing_site_x)
        self._hud_text(f"{downrange:08.1f} m", y, self.font_dsky_medium, DSKY_GREEN)
        y += y_inc_large

        # ── RCS ──
        self._hud_text("RCS", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += y_inc_small
        rcs_str = "OFF"
        if rcs_cmd < 0:
            rcs_str = "L"
        elif rcs_cmd > 0:
            rcs_str = "R"
        self._hud_text(rcs_str, y, self.font_dsky_medium, DSKY_GREEN)
        y += y_inc_large

        self._hud_separator(y)
        y += 8

        self._hud_text("DPS THROTTLE", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 14
        throttle_pct = int(throttle * 100)
        throttle_str = f"{throttle_pct:3d} %"
        self._hud_text(throttle_str, y, self.font_dsky_medium, DSKY_GREEN)
        
        bar_w = 140
        bar_h = 12
        bar_x = HUD_PADDING + 75
        bar_y = y + 5
        
        pygame.draw.rect(self.hud_surface, DSKY_DIM_GREEN, (bar_x, bar_y, bar_w, bar_h), 1)
        f_start = int(bar_w * 0.6)
        pygame.draw.rect(self.hud_surface, (100, 30, 30), (bar_x + f_start, bar_y + 1, bar_w - f_start, bar_h - 2))
        
        fill_w = int(bar_w * throttle)
        if fill_w > 0:
            pygame.draw.rect(self.hud_surface, WARNING_AMBER, (bar_x, bar_y + 1, fill_w, bar_h - 2))
        y += 24
        
        self._hud_text("PROPELLANT", y, self.font_dsky_title, DSKY_DIM_GREEN)
        
        if fuel_warning and (pygame.time.get_ticks() % 1000) < 500:
            self._hud_text("LOW FUEL", y, self.font_dsky_title, CRASHED_RED, x_offset=HUD_WIDTH - 90)
            
        y += 14
        prop_pct = (propellant / propellant_max * 100) if propellant_max > 0 else 0
        prop_str = f"{propellant:7.1f} kg"
        self._hud_text(prop_str, y, self.font_dsky_medium, DSKY_GREEN)
        self._hud_text(f"{prop_pct:3.0f}%", y, self.font_dsky_medium, DSKY_GREEN, x_offset=HUD_WIDTH - 65)
        y += 20
        
        pbar_w = HUD_WIDTH - 2 * HUD_PADDING
        pbar_h = 8
        pygame.draw.rect(self.hud_surface, DSKY_DIM_GREEN, (HUD_PADDING, y, pbar_w, pbar_h), 1)
        pfill_w = int(pbar_w * (propellant / propellant_max)) if propellant_max > 0 else 0
        
        p_color = DSKY_GREEN
        if prop_pct <= 10:
            p_color = CRASHED_RED
        elif prop_pct <= 20:
            p_color = WARNING_AMBER
            
        if pfill_w > 0:
            pygame.draw.rect(self.hud_surface, p_color, (HUD_PADDING, y + 1, pfill_w, pbar_h - 2))
        y += 14
        
        self._hud_text("DELTA-V", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 14
        self._hud_text(f"{delta_v:6.1f} m/s", y, self.font_dsky_medium, DSKY_GREEN)
        y += 20
        
        self._hud_text("BURN TIME", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 14
        if math.isinf(burn_time):
            bt_str = "---:--"
        else:
            bt_mins = int(burn_time // 60)
            bt_secs = int(burn_time % 60)
            bt_str = f"{bt_mins:02d}:{bt_secs:02d}"
        self._hud_text(bt_str, y, self.font_dsky_medium, DSKY_GREEN)
        y += 24

        self._hud_separator(y)
        y += 8

        self._hud_text("MET", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 14
        total_secs = time_elapsed
        hrs = int(total_secs // 3600)
        mins = int((total_secs % 3600) // 60)
        secs = total_secs % 60
        met_str = f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
        self._hud_text(met_str, y, self.font_dsky_medium, DSKY_GREEN)
        y += 24

        self._hud_separator(y)
        y += 8

        descent_rate = abs(velocity) if velocity < 0 else 0.0
        if status == "flying" and descent_rate > MAX_VERTICAL_SPEED:
            self._hud_text("⚠ HIGH DESCENT RATE", y, self.font_dsky_title, CRASHED_RED)
        elif status == "flying" and descent_rate > MAX_VERTICAL_SPEED * 0.7:
            self._hud_text("  DESCENT RATE", y, self.font_dsky_title, WARNING_AMBER)
        else:
            self._hud_text("  NOMINAL", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 18

        self._hud_text("STATUS", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 14
        status_colors = {
            "flying": DSKY_GREEN,
            "landed": LANDED_GREEN,
            "crashed": CRASHED_RED,
        }
        self._hud_text(
            status.upper(), y, self.font_dsky_medium,
            status_colors.get(status, DSKY_GREEN),
        )
        y += 24

        self._hud_separator(y)
        y += 8

        self._hud_text("── CONTROLS ──", y, self.font_dsky_title, DSKY_DIM_GREEN)
        y += 16
        controls = [
            ("A/D", "RCS Left/Right"),
            ("W/S", "Throttle Up/Down"),
            ("SPACE", "Toggle Engine"),
            ("P", "Pause"),
            ("R", "Reset"),
            ("F1", "Debug Overlay"),
            ("ESC", "Quit"),
        ]
        for key, action in controls:
            self._hud_text(f"{key:<10s}{action}", y, self.font_dsky_small, DSKY_DIM_GREEN)
            y += 14

        pygame.draw.rect(
            self.hud_surface, DSKY_BORDER,
            (0, 0, HUD_WIDTH, WINDOW_HEIGHT), 2,
        )

        pygame.draw.rect(
            self.hud_surface, (0, 80, 40),
            (4, 4, HUD_WIDTH - 8, WINDOW_HEIGHT - 8), 1,
        )

        self.screen.blit(self.hud_surface, (HUD_X, 0))

    def _hud_text(
        self,
        text: str,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x_offset: int = HUD_PADDING,
    ) -> None:
        surface = font.render(text, True, color)
        self.hud_surface.blit(surface, (x_offset, y))

    def _hud_separator(self, y: int) -> None:
        pygame.draw.line(
            self.hud_surface, DSKY_DIM_GREEN,
            (HUD_PADDING, y), (HUD_WIDTH - HUD_PADDING, y), 1,
        )

    def _draw_verdict(
        self, status: str, velocity: float, time_elapsed: float,
        propellant: float, delta_v: float, violations: list = None
    ) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        center_x = VIEW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2 - 30

        if status == "landed":
            title = "THE EAGLE HAS LANDED"
            color = LANDED_GREEN
            detail = f"Touchdown at {abs(velocity):.2f} m/s  (limit: 3.00 m/s)"
            margin = MAX_VERTICAL_SPEED - abs(velocity)
            extra = f"Margin: {margin:.2f} m/s  |  Time: {time_elapsed:.1f}s"
        else:
            title = "VEHICLE LOST"
            color = CRASHED_RED
            detail = f"Impact at {abs(velocity):.2f} m/s  (limit: 3.00 m/s)"
            excess = abs(velocity) - MAX_VERTICAL_SPEED
            extra = f"Exceeded by: {excess:.2f} m/s  |  Time: {time_elapsed:.1f}s"

        surf = self.font_verdict.render(title, True, color)
        rect = surf.get_rect(center=(center_x, center_y))
        self.screen.blit(surf, rect)

        # Print violations
        curr_y = center_y + 40
        if violations:
            for violation in violations:
                v_surf = self.font_info.render(violation, True, CRASHED_RED)
                v_rect = v_surf.get_rect(center=(center_x, curr_y))
                self.screen.blit(v_surf, v_rect)
                curr_y += 20
        else:
            surf = self.font_info.render(detail, True, WHITE)
            rect = surf.get_rect(center=(center_x, curr_y))
            self.screen.blit(surf, rect)
            curr_y += 25

        surf = self.font_info.render(extra, True, (180, 180, 180))
        rect = surf.get_rect(center=(center_x, curr_y))
        self.screen.blit(surf, rect)
        curr_y += 20
        
        extra2 = f"Fuel remaining: {propellant:.1f} kg  |  Delta-v used: {delta_v:.1f} m/s"
        surf2 = self.font_info.render(extra2, True, (180, 180, 180))
        rect2 = surf2.get_rect(center=(center_x, curr_y))
        self.screen.blit(surf2, rect2)
        curr_y += 40

        surf = self.font_dsky_small.render(
            "Press  R  to restart  |  ESC  to quit", True, DSKY_DIM_GREEN,
        )
        rect = surf.get_rect(center=(center_x, curr_y))
        self.screen.blit(surf, rect)

    def _draw_pause_overlay(self) -> None:
        overlay = pygame.Surface((VIEW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        center_x = VIEW_WIDTH // 2

        surf = self.font_verdict.render("║ PAUSED ║", True, DSKY_GREEN)
        rect = surf.get_rect(center=(center_x, WINDOW_HEIGHT // 2))
        self.screen.blit(surf, rect)

        surf = self.font_dsky_small.render(
            "Press P to resume", True, DSKY_DIM_GREEN,
        )
        rect = surf.get_rect(center=(center_x, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(surf, rect)

    def _draw_debug(
        self, altitude: float, velocity: float, thrust_on: bool,
        throttle: float, propellant: float, mass: float, x: float, vx: float, theta: float, omega: float, moi: float,
        agc_registers=None, guidance_mode: str = '',
        terrain=None, radar_locked: bool = False, radar_alt: float = 0.0
    ) -> None:
        """Draw physics debug information in the top-left corner."""
        from constants import G_MOON, DPS_MAX_THRUST

        thrust_accel = (DPS_MAX_THRUST * throttle) / mass if thrust_on else 0.0
        net_accel = thrust_accel - G_MOON
        twr = (DPS_MAX_THRUST * throttle) / (mass * G_MOON) if thrust_on else 0.0

        terrain_h = terrain.height_at(x) if terrain else 0.0

        debug_lines = [
            f"X P: {x:10.3f} m",
            f"ALT: {altitude:10.3f} m",
            f"V X: {vx:+10.3f} m/s",
            f"V Y: {velocity:+10.3f} m/s",
            f"ANG: {math.degrees(theta):+10.3f} deg",
            f"OMG: {math.degrees(omega):+10.3f} d/s",
            f"MAS: {mass:10.1f} kg",
            f"MOI: {moi:10.1f}",
            f"PRO: {propellant:10.1f} kg",
            f"THR: {throttle:10.3f}",
            f"ACC: {thrust_accel:10.3f} m/s²",
            f"GRV: {G_MOON:10.3f} m/s²",
            f"NET: {net_accel:+10.3f} m/s²",
            f"TWR: {twr:10.3f}",
            f"FPS: {self.clock.get_fps():10.1f}",
            f"RDR LCK: {radar_locked}",
            f"RDR ALT: {radar_alt:10.3f} m",
            f"TER HGT: {terrain_h:10.3f} m",
        ]
        
        if agc_registers:
            debug_lines.extend([
                "",
                f"MODE: {guidance_mode}",
                f"V/N: {agc_registers.verb:02d}/{agc_registers.noun:02d}",
                f"PROG ALM: {agc_registers.prog_light}"
            ])

        y = 10
        bg = pygame.Surface((220, len(debug_lines) * 16 + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.screen.blit(bg, (5, 5))

        for line in debug_lines:
            surf = self.font_debug.render(line, True, DSKY_GREEN)
            self.screen.blit(surf, (10, y))
            y += 16
