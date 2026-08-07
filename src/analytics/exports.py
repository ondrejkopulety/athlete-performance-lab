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

from config.settings import SUMMARIES_DIR
from src.db import repository as repo

log = logging.getLogger("analytics.exports")

CYCLING_SPORT_PATTERN = r"cycl|biking|ride"


def _write(df: pd.DataFrame, path: Path, label: str) -> int:
    if df.empty:
        log.warning("%s: žádná data, soubor nezapsán.", label)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("%s: %d řádků → %s", label, len(df), path)
    return len(df)


def export_activities(session: Session, path: Path | None = None) -> int:
    """Všechny aktivity včetně odvozených metrik."""
    path = path or SUMMARIES_DIR / "master_high_res_summary.csv"
    df = repo.read_activities(session, with_metrics=True)
    if not df.empty:
        df = df.sort_values("date")
    return _write(df, path, "Aktivity")


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


def export_all(session: Session) -> dict[str, int]:
    """Vygeneruje všechny standardní CSV exporty."""
    return {
        "activities": export_activities(session),
        "daily_metrics": export_daily_metrics(session),
        "cycling": export_cycling(session),
    }
