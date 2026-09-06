"""Phase 8 checks — the Saturn V must reach orbit with an honest ledger."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import ascent
from bodies import EARTH, G0


@pytest.fixture(scope="module")
def flight():
    return ascent.fly_to_orbit(dt=0.25, record=False)


class TestAscent:
    def test_reaches_stable_parking_orbit(self, flight):
        assert 170_000.0 < flight["perigee_alt"] < 220_000.0
        assert flight["apogee_alt"] < 260_000.0

    def test_orbital_speed_matches_visviva(self, flight):
        assert flight["v"] == pytest.approx(7_790.0, abs=60.0)

    def test_delta_v_ledger_is_coherent(self, flight):
        """ideal dv ~= orbital speed + gravity loss + drag loss (+ small
        steering/apsis residual). The gap between 7.8 and ~9.5 km/s IS
        the losses — this is the number the module exists to teach."""
        L = flight["losses"]
        assert 9_200.0 < L["ideal_dv"] < 9_900.0
        assert 900.0 < L["gravity"] < 1_400.0
        assert 60.0 < L["drag"] < 300.0
        residual = L["ideal_dv"] - (flight["v"] + L["gravity"] + L["drag"])
        assert abs(residual) < 800.0            # steering + Earth-rotation credit

    def test_sivb_keeps_tli_propellant(self, flight):
        """Enough S-IVB propellant must remain for a ~3.13 km/s TLI."""
        pl = flight["prop_left_sivb"]
        m0 = pl + ascent.S_IVB.dry + ascent.PAYLOAD
        ve = ascent.S_IVB.isp_vac * G0
        capability = ve * math.log(m0 / (m0 - pl))
        assert capability > 3_150.0
