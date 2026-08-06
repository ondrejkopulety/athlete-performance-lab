#!/usr/bin/env python3
"""
main.py – Garmin Training Analytics · CLI
==========================================

    python scripts/main.py                    # celá pipeline
    python scripts/main.py sync               # jen stažení z Garmin Connect
    python scripts/main.py load               # jen FIT soubory → databáze
    python scripts/main.py analyze            # jen přepočet metrik
    python scripts/main.py status             # co je v databázi

    python scripts/main.py analyze --force-metrics   # přepočítat i aktuální metriky
    python scripts/main.py load --force              # přeparsovat všechny FIT
    python scripts/main.py --skip-download           # celá pipeline bez sítě

Proti původní verzi zmizel krok `parse`: zapisoval CSV, které žádný další
krok nečetl, zatímco `merge` tytéž FIT soubory parsoval znovu. Dnes je to
jeden krok `load`, který zapisuje rovnou do databáze.

Vlastní logiku drží src/pipeline.py, aby CLI i API dělaly totéž.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import LOGS_DIR  # noqa: E402

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "main.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

from src.db.session import check_connection, session_scope  # noqa: E402


def _require_db() -> None:
    if not check_connection():
        log.error(
            "Databáze neodpovídá.\n"
            "  1) docker compose up -d db\n"
            "  2) .venv/bin/alembic upgrade head"
        )
        sys.exit(2)


def cmd_status() -> None:
    from sqlalchemy import func, select

    from src.db.models import Activity, DailyBiometrics, DailyMetrics, Record

    with session_scope() as session:
        counts = {
            "aktivity": session.scalar(select(func.count()).select_from(Activity)),
            "vteřinové záznamy": session.scalar(select(func.count()).select_from(Record)),
            "dny s biometrií": session.scalar(select(func.count()).select_from(DailyBiometrics)),
            "dny s metrikami": session.scalar(select(func.count()).select_from(DailyMetrics)),
        }
        last_act = session.scalar(select(func.max(Activity.date)))
        last_day = session.scalar(select(func.max(DailyMetrics.date)))

    print("\n  Databáze")
    print("  " + "─" * 44)
    for label, value in counts.items():
        print(f"  {label:<22} {value:>18,}".replace(",", " "))
    print(f"  {'poslední aktivita':<22} {str(last_act):>18}")
    print(f"  {'poslední den metrik':<22} {str(last_day):>18}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Garmin Training Analytics – pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "steps", nargs="*",
        choices=["sync", "load", "analyze", "status", "all"],
        default=["all"],
        help="Které kroky spustit (výchozí: all)",
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Vynech stahování z Garminu, pracuj s lokálními soubory")
    parser.add_argument("--force", action="store_true",
                        help="load: přeparsuj všechny FIT soubory bez ohledu na hash")
    parser.add_argument("--force-metrics", action="store_true",
                        help="analyze: přepočítej i aktivity s aktuální verzí metrik")
    parser.add_argument("--json", action="store_true",
                        help="Vypiš strojově čitelné shrnutí")
    args = parser.parse_args()

    steps = set(args.steps)
    if "status" in steps and len(steps) == 1:
        _require_db()
        cmd_status()
        return

    _require_db()
    report: dict = {}

    if "all" in steps:
        from src.pipeline import run_full_pipeline

        report = run_full_pipeline(
            skip_download=args.skip_download,
            force_load=args.force,
            force_activity_metrics=args.force_metrics,
        )
    else:
        if "sync" in steps:
            from src.pipeline import step_sync

            report["sync"] = step_sync()

        if "load" in steps:
            from src.ingestion.biometrics_import import import_biometrics
            from src.ingestion.loader import load_fit_files

            with session_scope() as session:
                report["biometrics_days"] = import_biometrics(session)
                result = load_fit_files(session, force=args.force)
                report["load"] = result.summary()

        if "analyze" in steps:
            from src.analytics.pipeline import run_analytics

            with session_scope() as session:
                result = run_analytics(session, force_activities=args.force_metrics)
                report["analytics"] = result.summary()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        log.info("Hotovo.")
        for key, value in report.items():
            log.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
