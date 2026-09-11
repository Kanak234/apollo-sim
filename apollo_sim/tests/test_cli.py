"""Tests for apollo-sim CLI interface."""
import pytest

from apollo_sim.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "apollo-sim 0.2.0" in captured.out or "apollo-sim 0.2.0" in captured.err


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Apollo Lunar Module" in captured.out


def test_cli_run_apollo11():
    ret = main(["--mission", "apollo11"])
    assert ret == 0


def test_cli_run_mars():
    ret = main(["--mission", "mars"])
    assert ret == 0
