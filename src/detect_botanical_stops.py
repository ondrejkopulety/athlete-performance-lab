"""Backwards-compatibility shim – real code lives in src/botanical/stops_detector.py."""
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.botanical.stops_detector import *  # noqa: F401,F403

if __name__ == "__main__":
    from src.botanical.stops_detector import main
    main()
