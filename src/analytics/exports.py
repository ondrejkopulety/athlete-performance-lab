"""
exports.py  –  CSV exporty z databáze
======================================

Databáze je zdroj pravdy, ale CSV se pořád hodí: otevřít v Excelu, poslat
někomu, prohnat vlastním skriptem. Tenhle modul je generuje **z databáze**,
takže mají vždy aktuální obsah.

Historicky ty soubory psala sama pipeline a po přechodu na databázi
přestaly být aktualizované — vypadaly aktuálně, ale byly zamrzlé. To je
horší, než kdyby chyběly.

Zmrazené kopie pro regresní testy žijí odděleně v data/_baseline/ a tenhle
modul na ně nesahá.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import SPLITS_DIR, SUMMARIES_DIR
from src.db import repository as repo

log = logging.getLogger("analytics.exports")

CYCLING_SPORT_PATTERN = r"cycl|biking|ride"

# Pořadí sloupců v cyklo splitech. Drží se tvaru, který psala původní
# scripts/legacy/split_cycling_activities.py, aby starší soubory ve složce
# a nově generované šly čist stejným kódem. `date` a `sport` v tabulce
# records nejsou – doplňují se z aktivity.
SPLIT_COLUMNS = [
    "activity_id", "timestamp", "date", "heart_rate", "speed", "distance",
    "altitude", "cadence", "power", "temperature", "vertical_oscillation",
    "stance_time", "respiratory_rate", "hrv", "position_lat", "position_long",
    "sport", "is_active", "hr_zone", "trimp_increment",
]


def _write(df: pd.DataFrame, path: Path, label: str) -> int:
    if df.empty:
        log.warning("%s: žádná data, soubor nezapsán.", label)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("%s: %d řádků → %s", label, len(df), path)
    return len(df)


def export_activities(session: Session, path: Path | None = None) -> int:
    """
    Všechny aktivity včetně odvozených metrik a tepové křivky.

    Tepová křivka se připojuje rozvinutá do sloupců ``hr_curve_5s`` …
    ``hr_curve_3600s`` – je jich pevný počet a jsou nezávislé na LTHR, takže
    se do řádku na aktivitu vejdou. Souvislé bloky sem nepatří: 11 prahů ×
    2 tolerance × 6 metrik je 132 hodnot na aktivitu a mají vlastní export
    v dlouhém formátu (``export_hr_blocks``).
    """
    path = path or SUMMARIES_DIR / "master_high_res_summary.csv"
    df = repo.read_activities(session, with_metrics=True)
    if df.empty:
        return _write(df, path, "Aktivity")

    curve = repo.read_hr_curve(session, wide=True)
    if not curve.empty:
        df = df.merge(curve, on="activity_id", how="left")

    return _write(df.sort_values("date"), path, "Aktivity")


def export_hr_blocks(session: Session, path: Path | None = None) -> int:
    """
    Souvislé bloky nad prahem – dlouhý formát, jeden řádek na práh × toleranci.

    Dlouhý formát schválně: je to tvar, ve kterém se to dotazuje ("kolik
    jsem letos vydržel v kuse nad 170") i tvar, který přežije změnu LTHR bez
    přepočtu. Rozvinutí do sloupců by znamenalo 132 sloupců na aktivitu.
    """
    path = path or SUMMARIES_DIR / "hr_blocks.csv"
    df = repo.read_hr_blocks(session)
    if not df.empty:
        df = df.drop(columns=["computed_at"], errors="ignore")
        df = df.sort_values(["activity_id", "threshold_bpm", "bridge_tolerance_s"])
    return _write(df, path, "Souvislé bloky")


def export_daily_metrics(session: Session, path: Path | None = None) -> int:
    """Denní metriky – nástupce athlete_readiness.csv."""
    path = path or SUMMARIES_DIR / "athlete_readiness.csv"
    df = repo.read_daily_metrics(session)
    if not df.empty:
        df = df.sort_values("date")
    return _write(df, path, "Denní metriky")


def export_cycling(
    session: Session,
    path: Path | None = None,
    since: date | None = None,
    until: date | None = None,
) -> int:
    """Jen cyklistické aktivity."""
    path = path or SUMMARIES_DIR / "cycling_summary.csv"
    df = repo.read_activities(session, since=since, until=until, with_metrics=True)
    if df.empty or "sport" not in df.columns:
        return _write(pd.DataFrame(), path, "Cyklistika")

    mask = df["sport"].str.contains(CYCLING_SPORT_PATTERN, case=False, na=False, regex=True)
    return _write(df[mask].sort_values("date"), path, "Cyklistika")


def _fmt(value: object, decimals: int = 0, default: str = "0") -> str:
    """Číslo do názvu souboru; chybějící hodnota nesmí shodit celý běh."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if pd.isna(f):
        return default
    return f"{f:.{decimals}f}"


def _split_filename(act: pd.Series) -> str:
    """2026_07_26_bike_62.7km_193min_1011m_23741293450.CSV"""
    date_part = str(act.get("date") or "0000-00-00")[:10].replace("-", "_")
    return (
        f"{date_part}_bike"
        f"_{_fmt(act.get('distance_km'), 1, '0.0')}km"
        f"_{_fmt(act.get('duration_minutes'))}min"
        f"_{_fmt(act.get('ascent_m'))}m"
        f"_{act['activity_id']}.CSV"
    )


def export_cycling_splits(
    session: Session,
    out_dir: Path | None = None,
    since: date | None = None,
    until: date | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Vteřinové záznamy každé cyklo aktivity jako samostatné CSV.

    Nahrazuje scripts/legacy/split_cycling_activities.py, který tytéž soubory
    skládal po kusech z 582MB master_high_res_training_data.csv – tedy z
    exportu, který pipeline po přechodu na databázi přestala aktualizovat.
    Zdrojem je teď tabulka records, takže výstup odpovídá stavu databáze.

    Běh je inkrementální: aktivita, která už soubor má, se přeskočí. Kompletní
    přepis znamená ~500 MB zápisu, takže se dělá jen na vyžádání (`force`).
    """
    out_dir = out_dir or SPLITS_DIR
    acts = repo.read_activities(session, since=since, until=until, with_metrics=True)
    result = {"written": 0, "skipped": 0, "no_records": 0}

    if acts.empty or "sport" not in acts.columns:
        log.warning("Cyklo splity: v databázi nejsou aktivity.")
        return result

    mask = acts["sport"].str.contains(CYCLING_SPORT_PATTERN, case=False, na=False, regex=True)
    cycling = acts[mask].sort_values("date")
    if cycling.empty:
        log.warning("Cyklo splity: žádná cyklistická aktivita.")
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Cyklo splity: %d aktivit → %s", len(cycling), out_dir)

    for _, act in cycling.iterrows():
        aid = str(act["activity_id"])
        # Garmin ID je celé číslo, takže podtržítko před ním stačí jako kotva:
        # kratší ID se nikdy neschová uvnitř delšího.
        existing = sorted(out_dir.glob(f"*_{aid}.CSV"))
        if existing and not force:
            result["skipped"] += 1
            continue

        df = repo.read_records(session, aid)
        if df.empty:
            log.debug("%s: aktivita bez vteřinových dat, přeskočeno.", aid)
            result["no_records"] += 1
            continue

        df["date"] = act["date"]
        df["sport"] = act["sport"]
        # Bez explicitního formátu píše pandas mezeru místo "T"; ISO tvar drží
        # nové soubory bajtově srovnatelné s těmi, co ve složce už leží.
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        target = out_dir / _split_filename(act)
        df.reindex(columns=SPLIT_COLUMNS).to_csv(target, index=False)
        # Název nese vzdálenost a převýšení. Když se přepočítaly, starý soubor
        # by tu zůstal jako druhá kopie téže aktivity.
        for stale in existing:
            if stale != target:
                stale.unlink()
        result["written"] += 1

    log.info(
        "Cyklo splity: zapsáno %d, přeskočeno %d, bez záznamů %d",
        result["written"], result["skipped"], result["no_records"],
    )
    return result


def export_all(session: Session) -> dict[str, int]:
    """
    Vygeneruje všechny standardní CSV exporty.

    Cyklo splity tu schválně nejsou: je to 400+ souborů a půl gigabajtu,
    což do každodenního běhu pipeline nepatří. Mají vlastní krok `splits`.
    """
    return {
        "activities": export_activities(session),
        "daily_metrics": export_daily_metrics(session),
        "cycling": export_cycling(session),
        "hr_blocks": export_hr_blocks(session),
    }
