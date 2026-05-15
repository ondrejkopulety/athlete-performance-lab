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
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import (
    DATA_DIR, RAW_DIR, FIT_DIR, SUMMARIES_DIR,
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR,
    CSV_HIGH_RES_SUMMARY, CSV_MASTER_SUMMARY,
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
# DATA FRESHNESS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _mtime(path: Path) -> float:
    """Return file mtime, or 0 if the file does not exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _fit_count() -> int:
    """Return the number of .fit files in FIT_DIR."""
    try:
        return sum(1 for f in FIT_DIR.iterdir() if f.suffix.lower() == ".fit")
    except OSError:
        return 0


def _load_dirty_ids() -> set:
    """Load dirty activity IDs from the marker file written by garmin_to_csv."""
    dirty_file = SUMMARIES_DIR / ".dirty_activity_ids.json"
    if dirty_file.exists():
        try:
            return set(json.loads(dirty_file.read_text()))
        except Exception:
            pass
    return set()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """Create all required data directories if they don't exist."""
    for d in (DATA_DIR, RAW_DIR, FIT_DIR, SUMMARIES_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Directory structure verified.")


def step_sync() -> tuple[bool, set]:
    """Step 1 – Download data from Garmin Connect API.

    Returns (success, dirty_activity_ids).
    success is True if sync completed (even partially), False on total failure.
    dirty_activity_ids contains IDs of activities that were synced/updated.
    """
    log.info("=" * 60)
    log.info("STEP 1 / 4 : SYNC – Garmin Connect API download")
    log.info("=" * 60)
    try:
        from src.api.garmin_client import main as sync_main
        dirty_ids = sync_main() or set()
        log.info(f"Sync returned {len(dirty_ids)} dirty activity ID(s).")
        return True, dirty_ids
    except ImportError:
        log.warning("garmin_to_csv not available – skipping sync.")
        return False, set()
    except SystemExit:
        # GarminRateLimitError causes sys.exit(1) inside garmin_to_csv
        log.error("Sync terminated due to rate limiting (HTTP 429). Pipeline continues with existing data.")
        # Try to recover dirty IDs from the marker file
        dirty_ids = _load_dirty_ids()
        return False, dirty_ids
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Garmin sync raised an unexpected error: {exc}. Continuing pipeline.")
        return False, set()


def step_parse() -> bool:
    """Step 2 – Parse FIT files to high-resolution CSV.

    Returns True if parse completed successfully.
    """
    log.info("=" * 60)
    log.info("STEP 2 / 4 : PARSE – FIT → High-Res CSV")
    log.info("=" * 60)
    try:
        from src.ingestion.fit_parser import main as parse_main
        parse_main()
        return True
    except ImportError:
        log.warning("fit_parser not available – skipping parse.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error(f"FIT parse failed: {exc}. Downstream steps may use stale data.")
        return False


def step_merge() -> bool:
    """Step 3 – Deduplicate & merge Garmin + Strava FIT data.

    Returns True if merge completed successfully.
    """
    log.info("=" * 60)
    log.info("STEP 3 / 4 : MERGE – Master Rebuild (deduplication)")
    log.info("=" * 60)
    try:
        from src.ingestion.master_rebuild import main as merge_main
        merge_main()
        return True
    except ImportError:
        log.warning("master_rebuild not available – skipping merge.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error(f"Master rebuild failed: {exc}. Analytics may use stale data.")
        return False


def step_analyze() -> bool:
    """Step 4 – Run analytics & generate readiness report.

    Returns True if analysis completed successfully.
    """
    log.info("=" * 60)
    log.info("STEP 4 / 5 : ANALYZE – Athlete Analytics")
    log.info("=" * 60)
    try:
        from src.analytics.athlete_analytics import main as analyze_main
        analyze_main()
        return True
    except ImportError:
        log.warning("athlete_analytics not available – skipping analysis.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error(f"Athlete analytics failed: {exc}")
        return False


def step_export_cycling() -> bool:
    """Step 5 – Export cycling activities to cycling_summary.csv.

    Returns True if export completed successfully.
    """
    log.info("=" * 60)
    log.info("STEP 5 / 5 : EXPORT CYCLING – Filtrování cyklistických aktivit")
    log.info("=" * 60)
    try:
        from src.analytics.export_cycling import main as export_cycling_main
        export_cycling_main()
        return True
    except ImportError:
        log.warning("export_cycling not available – skipping.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error(f"Cycling export failed: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = {
    "sync": step_sync,  # Returns tuple[bool, set]; called separately in main loop
    "parse": step_parse,
    "merge": step_merge,
    "analyze": step_analyze,
    "export_cycling": step_export_cycling,
}

# Steps whose failure should prevent downstream analytics from running
# on potentially stale data.  sync failure is tolerated (offline mode).
_DATA_PRODUCING_STEPS = {"parse", "merge"}


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force all steps even if upstream data unchanged",
    )
    args = parser.parse_args()

    log.info("Garmin Training Analytics  ·  Pipeline Start")
    log.info(f"Steps: {', '.join(args.steps)}")

    ensure_directories()

    t0 = time.time()
    results: dict[str, bool] = {}

    # ── Snapshot data state before pipeline ──────────────────────────────
    fit_count_before = _fit_count()
    parse_output = PROCESSED_DIR / CSV_HIGH_RES_SUMMARY
    merge_output = PROCESSED_DIR / CSV_MASTER_SUMMARY
    parse_mtime_before = _mtime(parse_output)
    merge_mtime_before = _mtime(merge_output)

    # Track whether upstream steps produced new data
    new_data_from_sync = False
    new_data_from_parse = False
    new_data_from_merge = False
    dirty_activity_ids: set = set()

    for name in args.steps:
        # If a data-producing step (parse/merge) failed earlier, warn but
        # still attempt subsequent steps – they may work with existing data.
        if name == "analyze":
            failed_upstream = [
                s for s in _DATA_PRODUCING_STEPS
                if s in results and not results[s]
            ]
            if failed_upstream:
                log.warning(
                    f"Upstream step(s) {', '.join(failed_upstream)} failed. "
                    f"Analytics will run on existing data (may be stale)."
                )
            # Skip analyze if no upstream data changed (unless --force or explicitly requested)
            if (
                not args.force
                and args.steps == list(STEPS.keys())  # full pipeline mode only
                and not new_data_from_parse
                and not new_data_from_merge
                and not dirty_activity_ids  # no dirty IDs from sync
                and merge_mtime_before > 0  # master CSV must already exist
            ):
                log.info("No new data from parse/merge – skipping analyze (use --force to override).")
                results[name] = True
                continue

        # ── Execute step ─────────────────────────────────────────────────
        if name == "sync":
            ok, dirty_activity_ids = step_sync()
        else:
            ok = STEPS[name]()
        results[name] = ok

        # ── Post-step freshness checks ──────────────────────────────────
        if name == "sync":
            new_fit_count = _fit_count()
            if new_fit_count > fit_count_before:
                new_data_from_sync = True
                log.info(f"Sync downloaded {new_fit_count - fit_count_before} new FIT file(s).")
            elif dirty_activity_ids:
                new_data_from_sync = True
                log.info(f"Sync reported {len(dirty_activity_ids)} dirty activity ID(s).")
            fit_count_before = new_fit_count

        elif name == "parse":
            new_mtime = _mtime(parse_output)
            if new_mtime > parse_mtime_before:
                new_data_from_parse = True
                log.info("Parse produced updated high-res data.")
            parse_mtime_before = new_mtime

        elif name == "merge":
            new_mtime = _mtime(merge_output)
            if new_mtime > merge_mtime_before:
                new_data_from_merge = True
                log.info("Merge produced updated master data.")
            merge_mtime_before = new_mtime

    elapsed = time.time() - t0

    # Summary
    failed = [s for s, ok in results.items() if not ok]
    log.info("=" * 60)
    if failed:
        log.warning(f"Pipeline finished in {elapsed:.1f} s  (failed steps: {', '.join(failed)})")
    else:
        log.info(f"Pipeline finished in {elapsed:.1f} s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
