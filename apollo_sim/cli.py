"""CLI entry point for apollo-sim."""
import argparse
import os
import sys

# Ensure package internal modules (ascent, bodies, etc.) resolve when imported as a package
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from . import run_apollo11, run_mars
except ImportError:
    import run_apollo11
    import run_mars


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="apollo-sim",
        description="Apollo Lunar Module and Mars Transfer Simulator",
    )
    parser.add_argument(
        "--mission",
        choices=["apollo11", "mars"],
        default="apollo11",
        help="Mission simulation to execute (default: apollo11)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save PNG plots for the mission trajectory",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="apollo-sim 0.2.0",
    )

    args = parser.parse_args(argv)

    if args.mission == "apollo11":
        run_apollo11.main(make_plots=args.plots)
    elif args.mission == "mars":
        run_mars.main(make_plots=args.plots)

    return 0


if __name__ == "__main__":
    sys.exit(main())
