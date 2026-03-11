#!/usr/bin/env python3
"""
WEED ANALYTICS PIPELINE v3.0 — Master Runner
==============================================
In-process orchestrator for the botanical pipeline:
  1) stops_detector   – Detect botanical stops + confidence scoring
  2) perf_analyzer    – Athletic performance before/after botanical stops
  3) hotspot_mapper   – DBSCAN clustering, geocoding & interactive map
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

from config.settings import LOGS_DIR, REPORTS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "weed.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("weed")

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🌿  W E E D   A N A L Y T I C S   P I P E L I N E  🌿    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_detect() -> None:
    """Step 1 – Detect botanical stops with physiological confidence scoring."""
    log.info("=" * 62)
    log.info("STEP 1 / 3 : DETECT – Botanical Stop Detection")
    log.info("=" * 62)
    from src.botanical.stops_detector import main as detect_main
    detect_main()


def step_performance() -> None:
    """Step 2 – Analyze athletic performance before/after botanical stops."""
    log.info("=" * 62)
    log.info("STEP 2 / 3 : PERFORMANCE – Before/After Analysis")
    log.info("=" * 62)
    from src.botanical.perf_analyzer import process_botanical_performance
    process_botanical_performance()


def step_hotspots() -> None:
    """Step 3 – DBSCAN clustering, geocoding & interactive map generation."""
    log.info("=" * 62)
    log.info("STEP 3 / 3 : HOTSPOTS – Clustering, Geocoding & Map")
    log.info("=" * 62)
    from src.botanical.hotspot_mapper import main as hotspot_main
    hotspot_main()


STEPS = {
    "detect": step_detect,
    "performance": step_performance,
    "hotspots": step_hotspots,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weed Analytics Pipeline — master runner"
    )
    parser.add_argument(
        "steps",
        nargs="*",
        choices=list(STEPS.keys()),
        default=list(STEPS.keys()),
        help="Pipeline step(s) to run (default: all)",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Skip FIT analysis, only run hotspot map generation.",
    )
    args = parser.parse_args()

    print(BANNER)

    if args.map_only:
        selected_steps = ["hotspots"]
        print("🗺  --map-only mode: running only clustering, geocoding & map generation.\n")
    else:
        selected_steps = args.steps

    total = len(selected_steps)
    pipeline_start = time.time()

    for i, name in enumerate(selected_steps, start=1):
        t0 = time.time()
        print(f"\n⏳  STEP {i}/{total}: {name}…")
        print("─" * 62)

        try:
            STEPS[name]()
        except SystemExit:
            pass  # individual scripts may call sys.exit(0) on success
        except Exception as exc:
            log.error("Step '%s' failed: %s", name, exc, exc_info=True)
            print(f"\n❌  Step {i} ({name}) failed: {exc}")
            print("    Pipeline interrupted — fix the error above and re-run.")
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"    ✔  Done in {elapsed:.1f} s")

    total_elapsed = time.time() - pipeline_start
    minutes, seconds = divmod(total_elapsed, 60)

    print("\n" + "═" * 62)
    print(f"✅  Pipeline v3.0 complete in {int(minutes)} min {seconds:.1f} s")
    print("📄  Interactive map: reports/hotspot_map.html")
    print("📊  Performance analysis: data/summaries/")
    print("═" * 62)


if __name__ == "__main__":
    main()
