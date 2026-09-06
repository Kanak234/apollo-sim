"""Run the full Apollo 11 mission chain and print a MET-stamped report.

    python3 run_apollo11.py [--plots]

Chains every phase module: Saturn V ascent -> TLI design -> LOI ->
powered descent (automated) -> ascent -> CW rendezvous -> TEI -> entry.
Prints a delta-v ledger against the historical values and, with
--plots, saves PNGs next to this file.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import ascent
import transfer
import descent_auto
import rendezvous as rz
import entry
import kepler
from bodies import EARTH, MOON, G0


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:03d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main(make_plots: bool = False) -> dict:
    print("=" * 72)
    print("APOLLO 11 — FULL MISSION SIMULATION (2D, patched conic)")
    print("=" * 72)

    # ── Phase 8: launch ─────────────────────────────────────────────
    fl = ascent.fly_to_orbit(dt=0.25, record=make_plots)
    met = fl["t"]
    print(f"\n[{hms(met)}] EARTH ORBIT INSERTION")
    print(f"    orbit {fl['perigee_alt']/1e3:.0f} x {fl['apogee_alt']/1e3:.0f} km,"
          f" v = {fl['v']:.0f} m/s")
    L = fl["losses"]
    print(f"    ideal dv {L['ideal_dv']:.0f}  = orbital {fl['v']:.0f}"
          f" + gravity {L['gravity']:.0f} + drag {L['drag']:.0f} + residual")

    # ── Phase 9: TLI ────────────────────────────────────────────────
    r_park = EARTH.radius + fl["perigee_alt"]
    d = transfer.design_tli(r_park)
    pl = fl["prop_left_sivb"]
    m0 = pl + ascent.S_IVB.dry + ascent.PAYLOAD
    ve = ascent.S_IVB.isp_vac * G0
    tli_capability = ve * math.log(m0 / (m0 - pl))
    met += 2.5 * 3600.0                       # ~1.5 orbits of checkout
    print(f"\n[{hms(met)}] TRANSLUNAR INJECTION")
    print(f"    dv required {d['tli_dv']:.0f} m/s | S-IVB capability"
          f" {tli_capability:.0f} m/s | margin"
          f" {tli_capability - d['tli_dv']:+.0f}")
    print(f"    arrival angle lambda = {d['lambda_deg']:.1f} deg, TOF"
          f" {d['tof_days']:.2f} days")

    # ── Phase 10: LOI ───────────────────────────────────────────────
    met += d["tof_days"] * 86_400.0
    loi1 = transfer.loi_delta_v(d["v_inf"])
    loi2 = transfer.circ_dv()
    print(f"\n[{hms(met)}] LUNAR ORBIT INSERTION")
    print(f"    perilune {d['perilune_alt']/1e3:.1f} km, v_inf"
          f" {d['v_inf']:.0f} m/s")
    print(f"    LOI-1 {loi1:.0f} m/s (Apollo 11: 889) | LOI-2 {loi2:.0f} m/s")

    # ── Powered descent (Phases 1-6 physics, automated) ────────────
    met += 20.0 * 3600.0
    de = descent_auto.fly_descent()
    print(f"\n[{hms(met)}] POWERED DESCENT  (automated P63-style)")
    print(f"    touchdown after {de['t']:.0f} s | v_vert"
          f" {de['v_vertical']:+.2f} m/s | v_horiz {de['v_horizontal']:.2f}")
    print(f"    dv used {de['dv_used']:.0f} m/s (Apollo 11: ~2,100) |"
          f" propellant left {de['prop_left']:.0f} kg")
    met += de["t"]

    # ── Phase 11: ascent + rendezvous ──────────────────────────────
    met += 21.6 * 3600.0                      # surface stay
    asc = rz.fly_ascent()
    print(f"\n[{hms(met)}] LUNAR ASCENT")
    print(f"    insertion {asc['perilune_alt']/1e3:.0f} x"
          f" {asc['apolune_alt']/1e3:.0f} km in {asc['t']:.0f} s |"
          f" dv {asc['dv_ideal']:.0f} m/s")
    n = math.sqrt(MOON.mu / (MOON.radius + 111_000.0) ** 3)
    r0, v0 = [-5_000.0, -20_000.0], [0.0, 0.0]
    T = 2.0 * math.pi / n
    dv1, dv2, tot = rz.two_impulse_intercept(r0, v0, n, T / 2.0)
    met += T / 2.0
    print(f"\n[{hms(met)}] RENDEZVOUS (CW two-impulse)")
    print(f"    dv1 {np.linalg.norm(dv1):.2f} + dv2 {np.linalg.norm(dv2):.2f}"
          f" = {tot:.2f} m/s")

    # ── TEI + entry ────────────────────────────────────────────────
    tei = transfer.tei_delta_v()
    met += 2.5 * 86_400.0
    print(f"\n[{hms(met)}] TRANS-EARTH INJECTION  dv {tei:.0f} m/s")
    en = entry.fly_entry(v_entry=11_032.0, gamma_deg=-6.5)
    met += en["t"]
    print(f"\n[{hms(met)}] ENTRY AND SPLASHDOWN")
    print(f"    outcome {en['outcome']} | peak {en['peak_g']:.1f} g |"
          f" splashdown {en['final_v']:.1f} m/s |"
          f" downrange {en['downrange_km']:.0f} km")

    print("\n" + "=" * 72)
    print("DELTA-V LEDGER (m/s)            simulated      historical")
    hist = {"Ascent (ideal)": (L["ideal_dv"], "~9,400"),
            "TLI": (d["tli_dv"], "3,050-3,200"),
            "LOI-1": (loi1, "889"),
            "Powered descent": (de["dv_used"], "~2,100"),
            "Lunar ascent": (asc["dv_ideal"], "~1,850-1,950"),
            "TEI": (tei, "~1,000 (with plane change)")}
    for k, (v, h) in hist.items():
        print(f"  {k:22s}  {v:10.0f}      {h}")
    print("=" * 72)

    if make_plots:
        _plots(fl, en)
    return {"ascent": fl, "tli": d, "descent": de, "entry": en}


def _plots(fl, en):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [h[0] for h in fl["history"]]
    alt = [ (math.hypot(h[1][0], h[1][1]) - EARTH.radius) / 1000.0
            for h in fl["history"]]
    v = [math.hypot(h[1][2], h[1][3]) for h in fl["history"]]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, alt, "b-", label="altitude [km]")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("altitude [km]", color="b")
    ax2 = ax1.twinx()
    ax2.plot(t, v, "r-", label="speed [m/s]")
    ax2.set_ylabel("speed [m/s]", color="r")
    ax1.set_title("Saturn V ascent — altitude and speed")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("ascent_profile.png", dpi=110)
    print("saved ascent_profile.png")


if __name__ == "__main__":
    main("--plots" in sys.argv)
