"""Backwards-compatibility shim – real code lives in src/analytics/master_rebuild.py."""
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.analytics.master_rebuild import *  # noqa: F401,F403
