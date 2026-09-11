"""Tests for Apollo 11 and Mars full-mission orchestrators."""
import apollo_sim.run_apollo11 as run_apollo11
import apollo_sim.run_mars as run_mars


def test_hms_formatting():
    assert run_apollo11.hms(0) == "000:00:00"
    assert run_apollo11.hms(3661) == "001:01:01"
    assert run_apollo11.hms(72000) == "020:00:00"


def test_apollo11_full_mission_simulation():
    res = run_apollo11.main(make_plots=False)
    assert isinstance(res, dict)
    assert "ascent" in res
    assert "tli" in res
    assert "descent" in res
    assert "entry" in res

    # Check key delta-v and telemetry values
    ascent_ideal = res["ascent"]["losses"]["ideal_dv"]
    tli_dv = res["tli"]["tli_dv"]
    descent_dv = res["descent"]["dv_used"]
    entry_outcome = res["entry"]["outcome"]
    entry_v = res["entry"]["final_v"]

    assert 9000 < ascent_ideal < 10000
    assert 3000 < tli_dv < 3400
    assert 2000 < descent_dv < 2400
    assert entry_outcome == "splashdown"
    assert entry_v < 10.0  # Parachute terminal velocity


def test_mars_full_mission_simulation():
    # Verify Mars mission simulation completes without errors
    run_mars.main(make_plots=False)
