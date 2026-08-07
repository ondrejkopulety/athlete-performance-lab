"""
apple_import.py  –  Apple Health → daily_biometrics
====================================================

Garmin je v provozu od srpna 2025, ale aktivity sahají do ledna 2022.
Apple Watch pokrývá 2019–2025, takže pro předgarminskou éru je to jediný
zdroj klidového tepu — a ten vstupuje do TRIMPu, tedy do celého PMC.

Parsování řeší `apple_health.py`, který už zvládá tři pasti Apple exportů:

  • datum je ve formátu DD.MM.YYYY (bez dayfirst=True se 158 dní naparsuje
    přehozeně, aniž by to cokoli ohlásilo)
  • spánek je textově „6h 6m", ne číslo
  • HRV je denní ROZSAH („40,19-75,58"), ne noční průměr

Ta poslední je důvod, proč se HRV **neimportuje**. Garmin `hrv_last_night`
je průměr RMSSD přes noc; Apple hodnota je rozpětí přes celý den. Vypadají
podobně, měří něco jiného, a smíchané by znehodnotily HRV baseline i
varovnou vlajku.

Zdroje se neslévají: řádky nesou `source='apple'` a čtenář si vybírá
prioritou (viz repo.read_biometrics_resolved).
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from sqlalchemy.orm import Session

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db import repository as repo
from src.ingestion.apple_health import (
    aggregate_daily,
    clean_raw,
    find_health_files,
    load_and_concat,
)

log = logging.getLogger("ingestion.apple")

SOURCE = "apple"

# Co z Apple exportu má protějšek v daily_biometrics.
# HRV chybí záměrně – viz docstring modulu.
COLUMN_MAP = {
    "resting_heart_rate": "resting_heart_rate",
    "duration_minutes": "sleep_duration_min",
    "vo2_max": "vo2_max",
    "steps": "steps",
}


def build_apple_biometrics() -> pd.DataFrame:
    """Načte Apple exporty a vrátí je ve tvaru tabulky daily_biometrics."""
    files = find_health_files()
    if not files:
        log.info("Žádné Apple exporty v data/Apple/ – přeskakuji.")
        return pd.DataFrame()

    daily = aggregate_daily(clean_raw(load_and_concat(files)))
    if daily.empty:
        log.warning("Apple exporty neobsahují použitelná denní data.")
        return pd.DataFrame()

    present = {src: dst for src, dst in COLUMN_MAP.items() if src in daily.columns}
    if "resting_heart_rate" not in present:
        log.warning("Apple data neobsahují klidový tep – import nemá co přinést.")
        return pd.DataFrame()

    out = daily[["date", *present.keys()]].rename(columns=present).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])

    for col in present.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "steps" in out.columns:
        out["steps"] = out["steps"].round().astype("Int64")

    # Řádky bez jediné hodnoty nemá smysl ukládat
    value_cols = list(present.values())
    out = out.dropna(subset=value_cols, how="all")

    out["source"] = SOURCE
    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def import_apple_biometrics(session: Session) -> int:
    """Uloží Apple biometrii do daily_biometrics. Vrací počet dní."""
    frame = build_apple_biometrics()
    if frame.empty:
        return 0

    n = repo.upsert_biometrics(session, repo.records_to_dicts(frame))
    rhr = frame["resting_heart_rate"].dropna()
    log.info(
        "Apple Health: %d dní (%s → %s), klidový tep u %d dní, medián %.0f bpm",
        n, frame["date"].min(), frame["date"].max(), len(rhr),
        rhr.median() if len(rhr) else float("nan"),
    )
    return n
