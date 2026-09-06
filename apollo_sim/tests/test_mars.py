"""Phases 13-15 checks: Lambert, windows, depots, ISRU, Mars EDL."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import marstransfer as mt
import marsops as mo
from bodies import EARTH, MARS, SUN


class TestWindowsAndHohmann:
    def test_synodic_period(self):
        assert mt.synodic_period_days() == pytest.approx(780.0, abs=2.0)

    def test_hohmann_reference(self):
        h = mt.hohmann_numbers()
        assert h["v_inf_dep"] == pytest.approx(2_945.0, abs=30.0)
        assert h["v_inf_arr"] == pytest.approx(2_649.0, abs=30.0)
        assert h["tof_days"] == pytest.approx(258.9, abs=2.0)
        assert h["phase_deg"] == pytest.approx(44.3, abs=1.0)

    def test_tmi_and_capture(self):
        h = mt.hohmann_numbers()
        tmi = mt.tmi_dv_from_leo(h["v_inf_dep"], EARTH.radius + 185_000.0)
        cap = mt.capture_dv_at_mars(h["v_inf_arr"], MARS.radius + 300_000.0)
        assert tmi == pytest.approx(3_616.0, abs=40.0)
        assert cap == pytest.approx(2_091.0, abs=40.0)


class TestLambert:
    def test_reproduces_near_hohmann(self):
        h = mt.hohmann_numbers()
        phase0 = math.radians(h["phase_deg"])
        tof = 255.0 * 86_400.0                 # near, not exactly, 180 deg
        rE, vE = mt.earth_state(0.0)
        rM, vM = mt.mars_state(tof, phase0)
        v1, v2 = mt.lambert(rE, rM, tof, SUN.mu)
        assert np.linalg.norm(v1 - vE) == pytest.approx(h["v_inf_dep"],
                                                        rel=0.02)
        assert np.linalg.norm(v2 - vM) == pytest.approx(h["v_inf_arr"],
                                                        rel=0.02)

    def test_porkchop_minimum_near_hohmann_total(self):
        h = mt.hohmann_numbers()
        phase0 = math.radians(h["phase_deg"])
        dep = np.arange(-48, 49, 16)
        tof = np.arange(180, 341, 20)
        pc = mt.porkchop(phase0, dep, tof, EARTH.radius + 185_000.0,
                         MARS.radius + 300_000.0)
        best = np.nanmin(pc)
        hoh_total = (mt.tmi_dv_from_leo(h["v_inf_dep"],
                                        EARTH.radius + 185_000.0)
                     + mt.capture_dv_at_mars(h["v_inf_arr"],
                                             MARS.radius + 300_000.0))
        assert best == pytest.approx(hoh_total, rel=0.03)


class TestDepotsAndIsru:
    def test_boiloff_math(self):
        m = mo.propellant_after_storage(100_000.0, 30.0, "LH2_passive")
        assert m == pytest.approx(100_000.0 * 0.99 ** 30, rel=1e-12)
        assert (mo.propellant_after_storage(1e5, 180, "CH4_passive")
                > mo.propellant_after_storage(1e5, 180, "LH2_passive"))

    def test_rocket_equation_with_structure(self):
        # 20 t payload through 3,616 m/s at Isp 450: hand value ~29 t.
        p = mo.stage_propellant_for_dv(20_000.0, 3_616.0, 450.0)
        assert p == pytest.approx(29_000.0, rel=0.03)

    def test_depot_shrinks_biggest_launch(self):
        a = mo.architecture_compare(20_000.0, 3_616.0, 450.0, 180.0,
                                    "LH2_cryocooled")
        assert a["depot_biggest_launch"] < 0.6 * a["direct_biggest_launch"]
        # ...but total IMLEO pays the boil-off penalty:
        assert a["depot_imleo"] > a["direct_imleo"]

    def test_isru_scaling(self):
        assert mo.isru_production_rate(100.0) == pytest.approx(160.0)
        assert mo.days_to_fill(30_000.0, 100.0) == pytest.approx(187.5)


class TestMarsEDL:
    def test_chute_alone_is_fatal(self):
        r = mo.fly_mars_edl(powered_final=False)
        assert r["outcome"] == "crashed"
        assert r["final_v"] > 40.0            # ~terminal on the chute

    def test_powered_final_lands(self):
        r = mo.fly_mars_edl(powered_final=True)
        assert r["outcome"] == "landed"
        assert r["final_v"] < 3.0
        assert 150.0 < r["burn_dv"] < 700.0

    def test_entry_heating_pulse_exists(self):
        r = mo.fly_mars_edl(powered_final=True)
        assert r["peak_g"] > 3.0              # real Mars entries: 5-10 g
