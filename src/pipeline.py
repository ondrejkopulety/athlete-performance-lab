"""
pipeline.py  –  celý běh na jednom místě
=========================================

Jediná definice toho, co znamená „aktualizuj data":

    1. SYNC    – stáhne novinky z Garmin Connect (CSV + FIT na disk)
    2. IMPORT  – biometrii z CSV do daily_biometrics
    3. LOAD    – nové/změněné FIT soubory do activities + records
    4. STRAVA  – doplní odkazy na aktivity na Stravě (activities.strava_id)
    5. ANALYZE – per-activity metriky (inkrementálně) + denní metriky
    6. HR      – tepová křivka, souvislé bloky a pokrytí (inkrementálně)
    7. EXPORT  – CSV z databáze, aby nezastarávaly pod rukama

Volá to CLI (scripts/main.py) i API (POST /api/sync/run), takže neexistují
dvě mírně odlišné verze pipeline, které se časem rozejdou.

Krok `parse` z původní pipeline zmizel – zapisoval CSV, které nikdo nečetl,
a `merge` tytéž FIT soubory parsoval znovu. Dnes je to jeden krok LOAD.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from config.settings import HR_COVERAGE_WARN_PCT
from src.analytics.exports import export_all
from src.analytics.pipeline import run_analytics
from src.db.session import session_scope
from src.ingestion.biometrics_import import import_biometrics
from src.ingestion.loader import load_fit_files

log = logging.getLogger("pipeline")


def step_hr(session) -> dict[str, Any]:
    """
    Předvýpočet tepové křivky, souvislých bloků a pokrytí.

    Běží po ANALYZE, protože čte ``records`` až po sloučení fragmentů a
    kanonizaci sportu. Bez tohohle kroku by dashboard u čerstvě
    naimportovaných jízd ukazoval prázdnou křivku a chybějící pokrytí –
    dřív se to spouštělo ručně přes ``python -m src.physio.cli hr``.

    Inkrementálně: aktivity, které už mají řádky v aktuální ``calc_version``,
    se přeskočí. Bump verze v settings tedy přepočet vynutí sám.
    """
    # Importy v těle kroku jsou záměrné: ``step_sync`` tahá ``garminconnect``
    # (těžké, nemusí být nainstalované u ``--skip-download``), ``step_hr``
    # zas ``src.physio`` batch. Držíme je stranou modulového importu, aby CLI
    # pro nesouvisející příkazy startovalo rychle a nepadalo na chybějící
    # volitelné závislosti.
    from sqlalchemy import select

    from config.settings import HR_BLOCKS_VERSION, HR_CURVE_VERSION
    from src.db import repository as repo
    from src.db.models import Activity, ActivityHrBlocks, ActivityHrCurve
    from src.physio.hr_batch import run_batch as run_hr_batch
    from src.physio.persist import write_hr_rows

    candidates = [
        row[0] for row in session.execute(select(Activity.activity_id)).all()
    ]
    done = repo.hr_computed_ids(session, ActivityHrBlocks, HR_BLOCKS_VERSION)
    # Křivka se posuzuje zvlášť, ale jen u aktivit, které nějaké řádky mají:
    # aktivita kratší než nejkratší okno je legitimně nemá a jinak by se
    # počítala při každém běhu znovu.
    done -= repo.hr_computed_ids(session, ActivityHrCurve, 0) - repo.hr_computed_ids(
        session, ActivityHrCurve, HR_CURVE_VERSION
    )
    todo = [a for a in candidates if a not in done]
    if not todo:
        return {"processed": 0, "cached": len(candidates)}

    series = repo.read_hr_series(session, todo)
    result = run_hr_batch(series, todo)
    written = write_hr_rows(
        session,
        result.curve_rows,
        result.block_rows,
        [c.to_row(HR_CURVE_VERSION) for c in result.coverage],
    )
    return {
        "processed": len(result.processed),
        "cached": len(candidates) - len(todo),
        "skipped_no_hr": len(result.skipped_no_hr),
        "low_coverage": len(result.low_coverage(HR_COVERAGE_WARN_PCT)),
        **written,
    }


def step_strava_map(session) -> dict[str, Any]:
    """
    Doplnění odkazů na aktivity na Stravě (``activities.strava_id``).

    Deduplikace zvládne spárovat jen jízdy, které mají Strava FIT na disku;
    novější jízdy odkaz získají tady, spárováním se seznamem aktivit z REST
    API Stravy podle času startu. Viz ``src/ingestion/strava_map.py``.

    Selhání (chybí OAuth v ``.env``, síť, rate limit) není fatální – jen se
    zaloguje, stejně jako u SYNC.
    """
    from src.ingestion.strava_map import StravaAuthError, map_strava_ids

    try:
        res = map_strava_ids(session)
        return {
            "ok": True,
            "updated": res.updated,
            "already_linked": res.already_linked,
            "unmatched": res.unmatched,
            "strava_activities": res.strava_activities,
        }
    except StravaAuthError as exc:
        log.warning("Strava párování přeskočeno: %s", exc)
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.warning("Strava párování selhalo (%s) – pokračuji.", exc)
        return {"ok": False, "error": str(exc)}


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

        if skip_download:
            log.info("Krok STRAVA přeskočen (--skip-download).")
            report["strava_map"] = {"skipped": True}
        else:
            log.info("── STRAVA ── odkazy na aktivity na Stravě")
            report["strava_map"] = step_strava_map(session)

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

        log.info("── HR ── tepová křivka, souvislé bloky, pokrytí")
        report["hr"] = step_hr(session)

        log.info("── EXPORT ── CSV z databáze")
        report["export"] = export_all(session)

    report["duration_s"] = round((datetime.now() - t0).total_seconds(), 1)
    log.info("Pipeline hotová za %.1f s", report["duration_s"])
    return report
