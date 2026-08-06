#!/usr/bin/env python3
"""
export_cycling.py – export cyklistických aktivit do CSV
========================================================
Vytáhne z databáze cyklistické aktivity (sport obsahuje cycl | biking | ride)
včetně odvozených metrik a uloží je do data/summaries/cycling_summary.csv.

Zdrojem je nově **databáze**, ne master_high_res_summary.csv. To CSV už
pipeline neaktualizuje, takže by skript tiše produkoval zastaralá data.

    python src/analytics/export_cycling.py
    python src/analytics/export_cycling.py --from 2026-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from config.settings import SUMMARIES_DIR  # noqa: E402
from src.db import repository as repo  # noqa: E402
from src.db.session import session_scope  # noqa: E402

CYCLING_CSV = SUMMARIES_DIR / "cycling_summary.csv"

# Pokryje cycling, cycling/road, cycling/gravel_cycling, biking, ride, …
SPORT_PATTERN = r"cycl|biking|ride"

log = logging.getLogger(__name__)


def export_cycling(
    dst: Path = CYCLING_CSV,
    since: date | None = None,
    until: date | None = None,
) -> int:
    """Zapíše cyklistické aktivity do *dst*. Vrací počet řádků."""
    with session_scope() as session:
        df = repo.read_activities(session, since=since, until=until, with_metrics=True)

    if df.empty:
        log.warning("Databáze neobsahuje žádné aktivity.")
        return 0
    if "sport" not in df.columns:
        log.error("Sloupec 'sport' chybí – export přerušen.")
        return 0

    mask = df["sport"].str.contains(SPORT_PATTERN, case=False, na=False, regex=True)
    cycling = df[mask].copy()
    if cycling.empty:
        log.warning("Žádné cyklistické aktivity nenalezeny.")
        return 0

    cycling = cycling.sort_values("date")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cycling.to_csv(dst, index=False)
    log.info("Exportováno %d cyklistických aktivit → %s", len(cycling), dst)
    return len(cycling)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cyklistických aktivit do CSV")
    parser.add_argument("--from", dest="since", type=date.fromisoformat, default=None)
    parser.add_argument("--to", dest="until", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=CYCLING_CSV)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    n = export_cycling(dst=args.out, since=args.since, until=args.until)
    if not n:
        sys.exit(1)


if __name__ == "__main__":
    main()
