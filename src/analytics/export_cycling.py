#!/usr/bin/env python3
"""
export_cycling.py – Garmin Training Analytics
==============================================
Reads  data/summaries/master_high_res_summary.csv,
filters cycling activities (sport contains: cycl | biking | ride),
and writes the result to data/summaries/cycling_summary.csv.

Single Responsibility: this script does ONE thing – cycling export.

Usage (standalone)
------------------
    python src/analytics/export_cycling.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402 – must come after sys.path setup

from config.settings import SUMMARIES_DIR  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_CSV  = SUMMARIES_DIR / "master_high_res_summary.csv"
CYCLING_CSV = SUMMARIES_DIR / "cycling_summary.csv"

# Case-insensitive regex matching sport values such as:
#   cycling, cycling/road, cycling/gravel_cycling, cycling/indoor_cycling,
#   biking, ride, …
SPORT_PATTERN = r"cycl|biking|ride"

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def export_cycling(
    src: Path = MASTER_CSV,
    dst: Path = CYCLING_CSV,
) -> int:
    """Filter cycling rows from *src* and write them to *dst*.

    Parameters
    ----------
    src : Path
        Path to the master summary CSV (must contain a ``sport`` column).
    dst : Path
        Destination path for the cycling-only CSV.

    Returns
    -------
    int
        Number of rows written (0 on error or empty result).
    """
    if not src.exists():
        log.error("Zdrojový soubor nenalezen: %s", src)
        return 0

    df = pd.read_csv(src, dtype=str, low_memory=False)

    if "sport" not in df.columns:
        log.error("Sloupec 'sport' nenalezen v %s – export přerušen.", src)
        return 0

    mask = df["sport"].str.contains(SPORT_PATTERN, case=False, na=False, regex=True)
    cycling_df = df[mask].copy()

    n = len(cycling_df)
    if n == 0:
        log.warning("Žádné cyklistické aktivity nebyly nalezeny v %s.", src)
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    cycling_df.to_csv(dst, index=False)
    log.info("Exportováno %d cyklistických aktivit → %s", n, dst)
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("=" * 60)
    log.info("EXPORT CYCLING – Filtrování cyklistických aktivit")
    log.info("  src : %s", MASTER_CSV)
    log.info("  dst : %s", CYCLING_CSV)
    log.info("=" * 60)

    n = export_cycling()

    log.info("=" * 60)
    if n:
        log.info("Hotovo – %d řádků uloženo do cycling_summary.csv", n)
    else:
        log.warning("Export neskončil úspěšně (viz výpisy výše).")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
