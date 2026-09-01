"""
context.py  –  strukturovaný kontext pro AI trenéra
====================================================

Vrací JSON, který jde vložit LLM do system promptu. Návrh se řídí třemi
věcmi, které se u „prostě mu pošleme čísla" typicky pokazí:

  1. **Jednotky a směr.** Model bez kontextu nepozná, že u hrv_cv_pct je
     nižší lepší nebo že ACWR má sweet spot uprostřed škály, ne na kraji.
     Proto se přikládá `metric_glossary` generovaný z jediné definice
     v config.settings.METRIC_META.

  2. **Logické pořadí.** Profil atleta → dnešek → trendy → poslední
     aktivity. Model čte odshora a první informace váží nejvíc.

  3. **Žádné NaN.** pandas NaN není platný JSON; všechno se serializuje
     jako null, jinak parser na druhé straně spadne.

Struktura je zároveň stabilní napříč dny, takže statickou část
(profil + glosář) lze v Anthropic API cachovat přes prompt caching a
platit plnou cenu jen za měnící se ocas.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config.settings import (
    ACWR_ACUTE_DAYS,
    ACWR_CHRONIC_DAYS,
    ATL_DAYS,
    CTL_DAYS,
    MAX_HR,
    METRIC_META,
    RESTING_HR,
    ZONES,
)
from src.analytics.calendar import today_local
from src.analytics.load import recommend
from src.db import repository as repo

# Kolik posledních aktivit přiložit. Víc než pár jich model stejně
# nevyužije a jen to nafoukne prompt.
RECENT_ACTIVITY_COUNT = 7
TREND_WINDOWS = (7, 28)

TODAY_SECTIONS: dict[str, list[str]] = {
    "training_load": ["ctl", "atl", "tsb", "trimp"],
    "recovery": [
        "pure_recovery_score", "readiness_score",
        "hrv_last_night", "hrv_weekly_avg", "hrv_cv_pct",
        "rhr_day", "rhr_baseline_14d", "rhr_baseline_90d", "rhr_elevation_bpm",
        "avg_stress_day",
        "sleep_score_day", "sleep_duration_min",
        "sleep_need_min", "sleep_performance_pct",
        "recovery_time_h", "garmin_readiness_score", "garmin_hrv_factor_pct",
    ],
    "risk": ["acwr", "ctl_ramp_rate", "illness_warning", "stress_flag_count"],
    "performance": ["lthr_estimate"],
    "quality": [
        "monotony", "strain", "whoop_strain",
        "polarization_low_pct", "polarization_high_pct", "z3_junk_pct",
        "fatigue_index",
    ],
    "metabolism": [
        "epoc_score_daily",
        "fat_g_daily", "carb_g_daily", "fluid_loss_l_daily",
    ],
}

TREND_METRICS = [
    "trimp", "ctl", "atl", "tsb", "acwr",
    "pure_recovery_score", "readiness_score",
    "hrv_last_night", "rhr_day", "rhr_baseline_90d", "lthr_estimate",
    "recovery_time_h",
    "sleep_duration_min",
    "sleep_score_day", "avg_stress_day", "monotony",
]

ACTIVITY_FIELDS = [
    "activity_id", "date", "sport", "activity_name", "duration_minutes",
    "total_trimp", "distance_km", "ascent_m", "avg_hr", "max_hr",
    "time_in_z1", "time_in_z2", "time_in_z3", "time_in_z4", "time_in_z5",
    "cardiac_drift", "max_hrr_60s", "durability_pct", "vam_m_per_h",
    "epoc_score", "avg_gradient_pct", "climb_category",
]  # resp_rate_rsa vynecháno – experimentální, viz METRIC_META


def _clean(value: Any) -> Any:
    """Cokoli z pandas → hodnota bezpečná pro json.dumps (NaN → None)."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value or None
    return value


def _section(row: pd.Series, keys: list[str]) -> dict[str, Any]:
    return {k: _clean(row.get(k)) for k in keys if k in row.index}


def _trend(df: pd.DataFrame, days: int) -> dict[str, Any]:
    """Průměr, minimum a maximum metrik za posledních N dní."""
    window = df.tail(days)
    if window.empty:
        return {}
    out: dict[str, Any] = {"days": days, "from": _clean(window.index.min()),
                           "to": _clean(window.index.max())}
    for metric in TREND_METRICS:
        if metric not in window.columns:
            continue
        series = pd.to_numeric(window[metric], errors="coerce").dropna()
        if series.empty:
            continue
        out[metric] = {
            "avg": _clean(series.mean()),
            "min": _clean(series.min()),
            "max": _clean(series.max()),
        }
    # Součet TRIMP dává smysl uvádět zvlášť – průměr sám o sobě neřekne,
    # kolik práce se za období skutečně udělalo.
    if "trimp" in window.columns:
        out["trimp_total"] = _clean(pd.to_numeric(window["trimp"], errors="coerce").sum())
        out["rest_days"] = int((pd.to_numeric(window["trimp"], errors="coerce").fillna(0) == 0).sum())
    return out


def _glossary(used_keys: set[str]) -> dict[str, Any]:
    """Glosář jen pro metriky, které se v kontextu opravdu vyskytly."""
    return {
        key: {k: v for k, v in meta.items() if v is not None}
        for key, meta in METRIC_META.items()
        if key in used_keys
    }


def build_daily_context(session: Session, day: date | None = None) -> dict[str, Any]:
    """
    Kompletní kontext pro daný den (výchozí: dnešek).

    Vrací čistý dict – serializovatelný, bez NaN, se stabilním pořadím klíčů.
    """
    day = day or today_local()

    daily = repo.read_daily_metrics(session, until=day)
    if daily.empty:
        return {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "date": day.isoformat(),
            "error": "V databázi nejsou žádné denní metriky.",
        }

    daily = daily.set_index(pd.to_datetime(daily["date"])).drop(columns=["date"])
    daily.index.name = "date"

    if pd.Timestamp(day) in daily.index:
        today_row = daily.loc[pd.Timestamp(day)]
        actual_day = day
    else:
        today_row = daily.iloc[-1]
        actual_day = daily.index[-1].date()

    # ── Dnešek po sekcích ──────────────────────────────────────────────────
    today: dict[str, Any] = {}
    used_keys: set[str] = set()
    for section, keys in TODAY_SECTIONS.items():
        today[section] = _section(today_row, keys)
        used_keys.update(keys)

    today["assessment"] = {
        "coach_advice": _clean(today_row.get("coach_advice")),
        "stress_flags": _clean(today_row.get("stress_flags")),
        "tsb_recommendation": recommend(
            pd.to_numeric(pd.Series([today_row.get("tsb")]), errors="coerce").iloc[0]
        ),
    }

    # ── Poslední aktivity ──────────────────────────────────────────────────
    activities = repo.read_activities(session, until=actual_day, with_metrics=True)
    recent: list[dict[str, Any]] = []
    if not activities.empty:
        activities = activities.sort_values("date").tail(RECENT_ACTIVITY_COUNT)
        for _, row in activities.iterrows():
            recent.append({f: _clean(row.get(f)) for f in ACTIVITY_FIELDS if f in row.index})
        recent.reverse()  # nejnovější první
        used_keys.update(ACTIVITY_FIELDS)

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "date": actual_day.isoformat(),
        "athlete": {
            "max_hr": MAX_HR,
            "resting_hr": RESTING_HR,
            "hr_zones_bpm": {z: list(bounds) for z, bounds in ZONES.items()},
            "model_windows": {
                "ctl_days": CTL_DAYS,
                "atl_days": ATL_DAYS,
                "acwr_acute_days": ACWR_ACUTE_DAYS,
                "acwr_chronic_days": ACWR_CHRONIC_DAYS,
            },
        },
        "today": today,
        "trends": {f"{d}d": _trend(daily, d) for d in TREND_WINDOWS},
        "recent_activities": recent,
        "metric_glossary": _glossary(used_keys),
        "notes": [
            "Všechny metriky pocházejí z hodinek Garmin a z FIT souborů; "
            "žádná hodnota není odhad modelu.",
            "TSB je posunuté o jeden den – reprezentuje ranní formu PŘED "
            "dnešním tréninkem.",
            "Chybějící hodnota (null) znamená, že měření pro daný den "
            "neexistuje – ne že je nulové.",
            "Dny bez tréninku mají trimp = 0; to je skutečná hodnota, ne chybějící údaj.",
        ],
    }


def build_history_series(
    session: Session,
    metric: str,
    days: int = 90,
    end: date | None = None,
) -> dict[str, Any]:
    """
    Časová řada jedné metriky – podklad pro budoucí tool use chatbota
    („ukaž mi HRV za poslední tři měsíce").
    """
    end = end or today_local()
    start = end - timedelta(days=days)
    daily = repo.read_daily_metrics(session, since=start, until=end)
    if daily.empty or metric not in daily.columns:
        return {"metric": metric, "points": [], "meta": METRIC_META.get(metric, {})}

    points = [
        {"date": _clean(r["date"]), "value": _clean(r[metric])}
        for _, r in daily.iterrows()
    ]
    return {
        "metric": metric,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "meta": METRIC_META.get(metric, {}),
        "points": points,
    }
