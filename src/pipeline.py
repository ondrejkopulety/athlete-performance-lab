"""
pipeline.py  –  celý běh na jednom místě
=========================================

Jediná definice toho, co znamená „aktualizuj data":

    1. SYNC    – stáhne novinky z Garmin Connect (CSV + FIT na disk)
    2. IMPORT  – biometrii z CSV do daily_biometrics
    3. LOAD    – nové/změněné FIT soubory do activities + records
    4. ANALYZE – per-activity metriky (inkrementálně) + denní metriky

Volá to CLI (scripts/main.py) i API (POST /api/sync/run), takže neexistují
dvě mírně odlišné verze pipeline, které se časem rozejdou.

Krok `parse` z původní pipeline zmizel – zapisoval CSV, které nikdo nečetl,
a `merge` tytéž FIT soubory parsoval znovu. Dnes je to jeden krok LOAD.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.analytics.pipeline import run_analytics
from src.db.session import session_scope
from src.ingestion.biometrics_import import import_biometrics
from src.ingestion.loader import load_fit_files

log = logging.getLogger("pipeline")


def step_sync() -> dict[str, Any]:
    """
    Stažení dat z Garmin Connect.

    Selhání není fatální – pipeline pokračuje nad tím, co je na disku.
    Výjimkou je rate limit (HTTP 429), který propaguje nahoru: pokračovat
    v dalších requestech by ban jen prodloužilo.
    """
    from src.ingestion.garmin_sync import GarminRateLimitError
    from src.ingestion.garmin_sync import main as garmin_main

    try:
        dirty_ids = garmin_main() or set()
        return {"ok": True, "dirty_activity_ids": len(dirty_ids)}
    except GarminRateLimitError as exc:
        log.error("Garmin rate limit – přerušuji stahování: %s", exc)
        return {"ok": False, "rate_limited": True, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.warning("Sync selhal (%s) – pokračuji s lokálními daty.", exc)
        return {"ok": False, "error": str(exc)}


def run_full_pipeline(
    skip_download: bool = False,
    force_load: bool = False,
    force_activity_metrics: bool = False,
) -> dict[str, Any]:
    """Spustí celý řetězec a vrátí strukturované shrnutí."""
    t0 = datetime.now()
    report: dict[str, Any] = {}

    if skip_download:
        log.info("Krok SYNC přeskočen (--skip-download).")
        report["sync"] = {"skipped": True}
    else:
        log.info("── SYNC ── stahování z Garmin Connect")
        report["sync"] = step_sync()

    with session_scope() as session:
        log.info("── IMPORT ── biometrie do databáze")
        report["biometrics_days"] = import_biometrics(session)

        log.info("── LOAD ── FIT soubory do databáze")
        load_result = load_fit_files(session, force=force_load)
        report["load"] = {
            "scanned": load_result.scanned,
            "inserted": load_result.inserted,
            "updated": load_result.updated,
            "skipped": load_result.skipped,
            "failed": load_result.failed,
            "records_written": load_result.records_written,
        }

        log.info("── ANALYZE ── metriky")
        analytics = run_analytics(session, force_activities=force_activity_metrics)
        report["analytics"] = {
            "activities_recomputed": analytics.activities_recomputed,
            "activities_total": analytics.activities_total,
            "days_written": analytics.days_written,
            "calendar_start": str(analytics.calendar_start),
            "calendar_end": str(analytics.calendar_end),
            "warnings": analytics.warnings[:20],
        }

    report["duration_s"] = round((datetime.now() - t0).total_seconds(), 1)
    log.info("Pipeline hotová za %.1f s", report["duration_s"])
    return report
