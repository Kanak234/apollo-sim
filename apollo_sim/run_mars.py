"""Run the Mars extension: windows, porkchop, EDL, ISRU, depot trade.

    python3 run_mars.py [--plots]
"""
from __future__ import annotations

import math
import sys

import marsops as mo
import marstransfer as mt
import numpy as np
from bodies import EARTH, MARS


def main(make_plots: bool = False):
    print("=" * 72)
    print("MARS EXTENSION — windows, transfer, EDL, ISRU, depots")
    print("=" * 72)

    print(f"\nLaunch window repeats every {mt.synodic_period_days():.0f} days"
          f" (the synodic period).")
    h = mt.hohmann_numbers()
    print(f"Hohmann reference: TOF {h['tof_days']:.0f} d, depart when Mars"
          f" leads by {h['phase_deg']:.1f} deg")
    r_leo = EARTH.radius + 185_000.0
    r_mo = MARS.radius + 300_000.0
    tmi = mt.tmi_dv_from_leo(h["v_inf_dep"], r_leo)
    cap = mt.capture_dv_at_mars(h["v_inf_arr"], r_mo)
    print(f"TMI from 185 km LEO: {tmi:.0f} m/s | capture to 300 km:"
          f" {cap:.0f} m/s | total {tmi + cap:.0f}")

    phase0 = math.radians(h["phase_deg"])
    dep = np.arange(-60, 61, 6)
    tof = np.arange(150, 391, 8)
    pc = mt.porkchop(phase0, dep, tof, r_leo, r_mo)
    i, j = np.unravel_index(np.nanargmin(pc), pc.shape)
    print(f"\nPorkchop minimum: {pc[i, j]:.0f} m/s at departure"
          f" {dep[i]:+d} d, TOF {tof[j]} d")

    print("\nMARS EDL")
    r1 = mo.fly_mars_edl(powered_final=False)
    print(f"  chute only : {r1['outcome']} at {r1['final_v']:.0f} m/s —"
          " the atmosphere is too thin to finish the job")
    r2 = mo.fly_mars_edl(powered_final=True)
    print(f"  powered    : {r2['outcome']} at {r2['final_v']:.1f} m/s,"
          f" landing burn {r2['burn_dv']:.0f} m/s, peak {r2['peak_g']:.1f} g")

    print("\nISRU (Sabatier, 15 kWh/kg assumption)")
    for kw in (10, 40, 100):
        print(f"  {kw:4d} kW -> {mo.isru_production_rate(kw):6.1f} kg/day,"
              f" 30 t ascent propellant in"
              f" {mo.days_to_fill(30_000, kw):6.0f} days")

    print("\nDEPOT TRADE (20 t payload through TMI, Isp 450, 180-day wait)")
    a = mo.architecture_compare(20_000.0, tmi, 450.0, 180.0,
                                "LH2_cryocooled")
    print(f"  direct : IMLEO {a['direct_imleo']/1e3:6.1f} t, biggest launch"
          f" {a['direct_biggest_launch']/1e3:6.1f} t")
    print(f"  depot  : IMLEO {a['depot_imleo']/1e3:6.1f} t, biggest launch"
          f" {a['depot_biggest_launch']/1e3:6.1f} t"
          f"  (boil-off cost {a['boiloff_penalty']/1e3:.1f} t)")
    print("  -> depots don't save total mass; they remove the need for a"
          " giant rocket.")

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        cs = ax.contourf(tof, dep, pc / 1000.0, levels=24, cmap="viridis")
        fig.colorbar(cs, label="total dv [km/s]")
        ax.plot(tof[j], dep[i], "r*", ms=14, label="minimum")
        ax.set_xlabel("time of flight [days]")
        ax.set_ylabel("departure offset [days]")
        ax.set_title("Earth->Mars porkchop (circular coplanar ephemerides)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("porkchop.png", dpi=110)
        print("\nsaved porkchop.png")


if __name__ == "__main__":
    main("--plots" in sys.argv)
