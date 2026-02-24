#!/usr/bin/env python3
"""
main.py – Garmin Training Analytics · Central Entry Point
==========================================================
Orchestrates the full pipeline:

  1. SYNC   – Download data from Garmin Connect API  (garmin_to_csv)
  2. PARSE  – Extract high-res training data from FIT files  (fit_to_highres_csv)
  3. MERGE  – Deduplicate Garmin + Strava FIT files  (master_rebuild)
  4. ANALYZE – Compute readiness, load management & physiology  (athlete_analytics)

Usage
-----
    python main.py                 # Run full pipeline
    python main.py sync            # Only download from Garmin Connect
    python main.py parse           # Only parse FIT → CSV
    python main.py merge           # Only deduplicate & merge
    python main.py analyze         # Only run analytics
    python main.py parse merge     # Chain specific steps
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import (
    DATA_DIR, RAW_DIR, FIT_DIR, SUMMARIES_DIR,
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR,
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "main.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """Create all required data directories if they don't exist."""
    for d in (DATA_DIR, RAW_DIR, FIT_DIR, SUMMARIES_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Directory structure verified.")


def step_sync() -> None:
    """Step 1 – Download data from Garmin Connect API."""
    log.info("=" * 60)
    log.info("STEP 1 / 4 : SYNC – Garmin Connect API download")
    log.info("=" * 60)
    mod = importlib.import_module("garmin_to_csv")
    if hasattr(mod, "main"):
        mod.main()
    else:
        log.warning("garmin_to_csv.main() not found – skipping sync.")


def step_parse() -> None:
    """Step 2 – Parse FIT files to high-resolution CSV."""
    log.info("=" * 60)
    log.info("STEP 2 / 4 : PARSE – FIT → High-Res CSV")
    log.info("=" * 60)
    mod = importlib.import_module("fit_to_highres_csv")
    if hasattr(mod, "main"):
        mod.main()
    else:
        log.warning("fit_to_highres_csv.main() not found – skipping parse.")


def step_merge() -> None:
    """Step 3 – Deduplicate & merge Garmin + Strava FIT data."""
    log.info("=" * 60)
    log.info("STEP 3 / 4 : MERGE – Master Rebuild (deduplication)")
    log.info("=" * 60)
    mod = importlib.import_module("master_rebuild")
    if hasattr(mod, "main"):
        mod.main()
    else:
        log.warning("master_rebuild.main() not found – skipping merge.")


def step_analyze() -> None:
    """Step 4 – Run analytics & generate readiness report."""
    log.info("=" * 60)
    log.info("STEP 4 / 4 : ANALYZE – Athlete Analytics")
    log.info("=" * 60)
    mod = importlib.import_module("athlete_analytics")
    if hasattr(mod, "main"):
        mod.main()
    else:
        log.warning("athlete_analytics.main() not found – skipping analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = {
    "sync": step_sync,
    "parse": step_parse,
    "merge": step_merge,
    "analyze": step_analyze,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Garmin Training Analytics – pipeline runner",
        epilog="Without arguments runs the full pipeline: sync → parse → merge → analyze",
    )
    parser.add_argument(
        "steps",
        nargs="*",
        choices=list(STEPS.keys()),
        default=list(STEPS.keys()),
        help="Pipeline step(s) to run (default: all)",
    )
    args = parser.parse_args()

    log.info("Garmin Training Analytics  ·  Pipeline Start")
    log.info(f"Steps: {', '.join(args.steps)}")

    ensure_directories()

    t0 = time.time()
    for name in args.steps:
        STEPS[name]()
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info(f"Pipeline finished in {elapsed:.1f} s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
