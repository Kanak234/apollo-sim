"""pytest configuration — adds lunar_lander/ to the import path."""
import sys
from pathlib import Path

# Add the lunar_lander directory to sys.path so tests can import
# physics, vehicle, constants, etc. without package prefixes.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
