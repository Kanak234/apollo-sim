"""Phases 9-10 checks against the historical Apollo numbers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import transfer
from bodies import EARTH


@pytest.fixture(scope="module")
def design():
    return transfer.design_tli(EARTH.radius + 190_000.0)


class TestLunarTransfer:
    def test_hohmann_tli_floor(self):
        """Minimum-energy TLI from 190 km is ~3,130 m/s; nothing real
        can be below it."""
        dv = transfer.tli_delta_v_hohmann(EARTH.radius + 190_000.0)
        assert dv == pytest.approx(3_131.0, abs=15.0)

    def test_designed_tli_close_to_apollo(self, design):
        # Apollo TLI burns were ~3,050-3,200 m/s depending on the
        # parking orbit; a fast 2-3 day patched-conic sits slightly hot.
        assert 3_100.0 < design["tli_dv"] < 3_300.0

    def test_perilune_targeting(self, design):
        assert design["perilune_alt"] == pytest.approx(111_000.0, abs=6_000.0)

    def test_time_of_flight_is_days_not_hours(self, design):
        assert 1.8 < design["tof_days"] < 3.6

    def test_loi_matches_apollo_11(self, design):
        # Apollo 11 LOI-1 was 889 m/s.
        dv = transfer.loi_delta_v(design["v_inf"])
        assert 800.0 < dv < 1_050.0

    def test_loi2_is_small(self):
        assert 20.0 < transfer.circ_dv() < 80.0

    def test_tei_bracket(self):
        # Apollo 11 TEI was ~1,000 m/s including plane change; the
        # planar ideal sits a bit lower.
        assert 750.0 < transfer.tei_delta_v() < 1_100.0

    def test_energy_bookkeeping_at_soi(self, design):
        """v_inf must be consistent with the selenocentric energy."""
        sel = design["arrival"]
        assert sel["energy"] > 0.0                 # hyperbolic arrival
        assert sel["v_inf"] == pytest.approx(
            (2.0 * sel["energy"]) ** 0.5, rel=1e-9)
