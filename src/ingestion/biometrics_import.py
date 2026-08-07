"""
biometrics_import.py  –  denní biometrie z Garmin CSV do databáze
==================================================================

`garmin_sync.py` zůstává tak, jak je: stahuje z Garmin Connect a ukládá
odpovědi do data/summaries/*.csv. Ta vrstva má vlastní ošetření rate
limitů, retry logiku a správu tokenů, kterou nelze otestovat jinak než
proti živému API – přepisovat ji naslepo by bylo riskantní (HTTP 429
znamená hodinový ban).

Tenhle modul je tedy seam mezi CSV a databází: po každém syncu přetaví
šest denních CSV do jedné tabulky daily_biometrics. Stejnou funkci volá
i jednorázová migrace historických dat, takže existuje jen jedna cesta,
kterou biometrie do DB vstupuje.

Poznámka ke spánku: Garmin reportuje fáze v procentech, DB drží minuty –
s minutami se lépe počítá i lépe odpovídá chatbotovi.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import SUMMARIES_DIR
from src.db import repository as repo

log = logging.getLogger("ingestion.biometrics")

# Zdrojové CSV → sloupce v daily_biometrics
SOURCES: dict[str, dict[str, str]] = {
    "hrv.csv": {
        "last_night_avg": "hrv_last_night",
        "weekly_avg": "hrv_weekly_avg",
    },
    "daily_health.csv": {
        "resting_heart_rate": "resting_heart_rate",
        "stress_average": "stress_average",
        "body_battery_highest": "body_battery_max",
        "body_battery_lowest": "body_battery_min",
    },
    "sleep.csv": {
        "sleep_score": "sleep_score",
        "duration_minutes": "sleep_duration_min",
        "deep_sleep_percentage": "_deep_pct",
        "light_sleep_percentage": "_light_pct",
        "rem_sleep_percentage": "_rem_pct",
    },
    "vo2_max.csv": {"vo2_max": "vo2_max"},
    "movement.csv": {"steps": "steps"},
    "intensity.csv": {"total_intensity_min": "intensity_minutes"},
}

PCT_TO_MINUTES = {
    "_deep_pct": "sleep_deep_min",
    "_light_pct": "sleep_light_min",
    "_rem_pct": "sleep_rem_min",
}


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.date


def _load(name: str, rename: dict[str, str]) -> pd.DataFrame:
    path = SUMMARIES_DIR / name
    if not path.exists():
        log.debug("%s neexistuje – přeskakuji.", name)
        return pd.DataFrame(columns=["date"])

    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns or df.empty:
        log.warning("%s nemá sloupec date nebo je prázdné.", name)
        return pd.DataFrame(columns=["date"])

    df["date"] = _to_date(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    present = {src: dst for src, dst in rename.items() if src in df.columns}
    return df[["date", *present.keys()]].rename(columns=present)


def build_biometrics_frame(since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Sloučí všechna denní CSV do jednoho DataFrame indexovaného datem."""
    merged: pd.DataFrame | None = None
    for name, rename in SOURCES.items():
        frame = _load(name, rename)
        if frame.empty or len(frame.columns) <= 1:
            continue
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    duration = pd.to_numeric(merged.get("sleep_duration_min"), errors="coerce")
    for pct_col, out_col in PCT_TO_MINUTES.items():
        if pct_col in merged.columns:
            merged[out_col] = (
                duration * pd.to_numeric(merged[pct_col], errors="coerce") / 100.0
            ).round(1)
    merged = merged.drop(columns=[c for c in PCT_TO_MINUTES if c in merged.columns])

    for col in [c for c in merged.columns if c != "date"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if "steps" in merged.columns:
        merged["steps"] = merged["steps"].round().astype("Int64")

    merged["source"] = "garmin"
    merged = merged.sort_values("date")
    if since is not None:
        merged = merged[pd.to_datetime(merged["date"]) >= since]
    return merged


def import_biometrics(session: Session, since: pd.Timestamp | None = None) -> int:
    """Načte denní biometrii z CSV do daily_biometrics. Vrací počet dní."""
    frame = build_biometrics_frame(since=since)
    if frame.empty:
        log.info("Žádná biometrická data k importu.")
        return 0

    n = repo.upsert_biometrics(session, repo.records_to_dicts(frame))
    log.info("Biometrie: %d dní (%s → %s)", n, frame["date"].min(), frame["date"].max())
    return n
