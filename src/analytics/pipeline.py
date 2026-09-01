"""
pipeline.py  –  orchestrace analytiky
======================================

Dva kroky s velmi odlišnou cenou:

  1. **Per-activity metriky** (drahé) – počítají se jen pro aktivity, kterým
     chybí `activity_metrics` nebo mají zastaralou `metrics_version`.
     Při denním běhu je to typicky jedna aktivita místo 864.

  2. **Denní metriky** (levné) – přepočítávají se vždy celé. Rolling okna,
     42denní EMA a ffill u ef_trend nesou vliv řádově rok zpátky, takže
     částečný přepočet by dal jiná čísla než plný. Nad 1600 dny to trvá
     desítky milisekund, takže není co optimalizovat.

Pořadí výpočtů odpovídá původnímu athlete_analytics.main() – jednotlivé
kroky na sebe navazují (readiness potřebuje spánek z recovery, fatigue_index
potřebuje efficiency, coach_advice potřebuje ACWR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config.settings import (
    ACTIVITY_METRICS_VERSION,
    DAILY_METRICS_VERSION,
    LTHR_FACTOR,
    LTHR_TEST_MINUTES,
    LTHR_WINDOW_DAYS,
    RESTING_HR,
)
from src.analytics import activity as act
from src.analytics import advice, biometrics, load, quality
from src.analytics.calendar import build_calendar
from src.db import repository as repo

log = logging.getLogger("analytics.pipeline")

# Sloupce activity_metrics, které pipeline zapisuje (rr_intervals_ms se
# záměrně nepřepisuje – patří ingest vrstvě).
ACTIVITY_METRIC_COLUMNS = [
    "trimp_adjusted", "rhr_used",
    "best_20min_hr", "best_30min_hr", "best_60min_hr",
    "cardiac_drift", "max_hrr_60s", "durability_pct", "vam_m_per_h",
    "avg_gradient_pct", "climb_category",
    "resp_rate_rsa", "epoc_score",
    # Diagnostiku R-R (dfa_quality, rr_*) plní src/physio/persist.py, ne
    # tahle pipeline – v seznamu proto není, aby ji přepočet nepřepsal na NULL.
    "time_at_threshold_min", "tte_z4z5_min",
    "fat_kcal", "carb_kcal", "fat_g",
    "carb_g", "fluid_loss_l", "heat_flag",
]

# Sloupce, které plní jen `compute_activity_series_metrics` (vteřinová data).
# U aktivit mimo `is_series_eligible` – typicky elektrokolo – se musí aktivně
# vynulovat: přepočet jinak ponechá staré hodnoty z předchozí verze, protože
# payload se skládá z už načteného `activities` DataFrame.
SERIES_ONLY_COLUMNS = [
    "best_20min_hr", "best_30min_hr", "best_60min_hr",
    "cardiac_drift", "max_hrr_60s", "durability_pct", "resp_rate_rsa",
]

# Metriky odvozené z celé historie, ne z jedné aktivity. Musí se
# přepisovat u všech řádků, jinak by se v tabulce míchaly různé škály.
GLOBAL_METRIC_COLUMNS = ["trimp_load_percentile"]

DAILY_METRIC_COLUMNS = [
    "trimp", "ctl", "atl", "tsb",
    "acwr", "ctl_ramp_rate", "ctl_ramp_warning",
    "monotony", "strain", "whoop_strain",
    "daily_efficiency", "ef_trend", "fatigue_index",
    "polarization_low_pct", "polarization_high_pct", "z3_junk_pct",
    "readiness_score", "pure_recovery_score",
    "hrv_last_night", "hrv_weekly_avg", "hrv_cv_pct",
    "rhr_day", "rhr_baseline_14d", "rhr_baseline_90d", "rhr_elevation_bpm",
    "rhr_source", "lthr_estimate", "avg_stress_day",
    "sleep_score_day", "sleep_duration_min", "sleep_need_min",
    "sleep_performance_pct", "max_hrr_60s_avg",
    "stress_flag_count", "illness_warning", "stress_flags", "coach_advice",
    "epoc_score_daily",
    "recovery_time_h", "garmin_readiness_score", "garmin_hrv_factor_pct",
    "fat_kcal_daily", "carb_kcal_daily", "fat_g_daily", "carb_g_daily",
    "fluid_loss_l_daily",
]

# Zaokrouhlení výstupu – drží stejný počet desetinných míst jako původní CSV.
ROUND_MAP = {
    "ctl": 2, "atl": 2, "tsb": 2,
    "monotony": 2, "strain": 1, "whoop_strain": 2,
    "daily_efficiency": 4, "ef_trend": 4,
    "readiness_score": 1, "max_hrr_60s_avg": 1,
    "pure_recovery_score": 1,
    "rhr_day": 0, "rhr_baseline_14d": 1, "rhr_baseline_90d": 1,
    "rhr_elevation_bpm": 1, "lthr_estimate": 0,
    "avg_stress_day": 0,
    "hrv_last_night": 1, "hrv_weekly_avg": 1,
    "sleep_score_day": 0, "sleep_duration_min": 0,
    "sleep_need_min": 0, "sleep_performance_pct": 1,
    "hrv_cv_pct": 1,
    "polarization_low_pct": 1, "polarization_high_pct": 1,
    "acwr": 2, "ctl_ramp_rate": 2,
    "fatigue_index": 3,
    "fat_kcal_daily": 0, "carb_kcal_daily": 0,
    "fat_g_daily": 0, "carb_g_daily": 0,
    "fluid_loss_l_daily": 2,
    "epoc_score_daily": 1,
    "recovery_time_h": 1, "garmin_readiness_score": 0, "garmin_hrv_factor_pct": 0,
}


@dataclass
class AnalyticsResult:
    activities_recomputed: int = 0
    activities_total: int = 0
    days_written: int = 0
    calendar_start: date | None = None
    calendar_end: date | None = None
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.activities_recomputed}/{self.activities_total} aktivit přepočítáno, "
            f"{self.days_written} dní zapsáno "
            f"({self.calendar_start} → {self.calendar_end}) za {self.duration_s:.1f} s"
        )


# ═══════════════════════════════════════════════════════════════════════════
# KROK 1 – per-activity metriky
# ═══════════════════════════════════════════════════════════════════════════

def _rhr_lookup(session: Session, activities: pd.DataFrame) -> pd.Series:
    """
    Klidový tep k datu — 90denní medián, zarovnaný na kalendář aktivit.

    Kde měření chybí (např. rok 2022, kdy ještě neexistoval ani Apple
    záznam), se použije konstanta ze settings. Ta v konfiguraci zůstává
    právě jako tahle záloha.
    """
    dates = pd.to_datetime(activities["date"])
    index = pd.date_range(dates.min(), dates.max(), freq="D")
    biometrics_df = repo.read_biometrics_resolved(session)
    baseline = biometrics.rhr_baseline_series(biometrics_df, index)
    return baseline.fillna(RESTING_HR)


def compute_activity_metrics(
    session: Session,
    force: bool = False,
    result: AnalyticsResult | None = None,
) -> pd.DataFrame:
    """
    Dopočítá odvozené metriky aktivit a uloží je do activity_metrics.

    Vrací kompletní tabulku aktivit včetně metrik (potřebuje ji denní část).
    """
    result = result or AnalyticsResult()

    activities = repo.read_activities(session, with_metrics=True)
    if activities.empty:
        log.warning("Databáze neobsahuje žádné aktivity.")
        return activities
    result.activities_total = len(activities)

    # Klidový tep platný k datu každé aktivity. Musí se načíst dřív než
    # per-activity metriky, protože z něj vychází přepočet TRIMP.
    rhr_lookup = _rhr_lookup(session, activities)

    stale = (
        activities["activity_id"].tolist()
        if force
        else repo.stale_activity_ids(session, ACTIVITY_METRICS_VERSION)
    )

    # ── Vteřinová data: jen pro zastaralé aktivity ─────────────────────────
    # Každá aktivita se čte právě jednou. TRIMP se přepočítává vždy (týká
    # se i chůze a posilovny), vteřinová fyziologie jen u kardio aktivit
    # nad 20 minut – u patnáctiminutové procházky nemá drift ani DFA smysl.
    series_rows: dict[str, dict] = {}
    if stale:
        meta = activities.set_index("activity_id")
        log.info("Per-activity metriky: %d zastaralých aktivit.", len(stale))

        for i, aid in enumerate(stale, 1):
            if aid not in meta.index:
                continue
            tdata = repo.read_records(session, aid)
            if tdata.empty:
                result.warnings.append(f"{aid}: chybí vteřinová data")
                continue

            act_date = pd.Timestamp(meta.at[aid, "date"])
            rhr = float(rhr_lookup.get(act_date, RESTING_HR))
            row = {
                "trimp_adjusted": act.compute_trimp_from_records(tdata, rhr),
                "rhr_used": round(rhr, 1),
            }

            if act.is_series_eligible(meta.at[aid, "sport"], meta.at[aid, "duration_minutes"]):
                rr = repo.read_rr_intervals(session, aid)
                row.update(
                    act.compute_activity_series_metrics(tdata, meta.at[aid, "sport"], rr)
                )
            else:
                # Elektrokolo a spol.: vyčisti fyziologii z předchozí verze,
                # ať v tabulce nezůstane drift/DFA/tepová křivka spočítaná
                # tehdy, když aktivita ještě mezi způsobilé patřila.
                row.update({col: None for col in SERIES_ONLY_COLUMNS})

            series_rows[aid] = row
            if i % 50 == 0 or i == len(stale):
                log.info("  [%d/%d] zpracováno", i, len(stale))
    else:
        log.info("Per-activity metriky jsou aktuální (verze %d).", ACTIVITY_METRICS_VERSION)

    # ── Vektorové metriky: vždy nad celou tabulkou ─────────────────────────
    # Critical HR je percentil napříč historií, takže se musí počítat ze
    # všech aktivit – ne jen z těch nově načtených.
    activities = act.compute_activity_table(activities)

    for aid, metrics in series_rows.items():
        for col, value in metrics.items():
            activities.loc[activities["activity_id"] == aid, col] = value

    # ── Zápis: jen přepočítané řádky (u ostatních by upsert jen přepsal
    #    tytéž hodnoty a zbytečně bumpnul metrics_version) ────────────────
    to_write = activities if force else activities[activities["activity_id"].isin(stale)]
    if not to_write.empty:
        payload = to_write[
            ["activity_id", *[c for c in ACTIVITY_METRIC_COLUMNS if c in to_write.columns]]
        ].copy()
        payload["metrics_version"] = ACTIVITY_METRICS_VERSION
        payload["computed_at"] = datetime.now()
        repo.upsert_activity_metrics(session, repo.records_to_dicts(payload))
        result.activities_recomputed = len(payload)

    # ── Globálně odvozené metriky se zapisují VŽDY pro všechny aktivity ───
    # trimp_load_percentile je pořadí napříč celou historií, takže každá nová
    # aktivita posune percentil i pro roky staré záznamy. Kdyby se zapisovaly
    # jen přepočítané řádky, zůstala by v tabulce směs starých a nových
    # percentilů a nešly by porovnávat napříč sezónami.
    if stale and not force:
        globals_payload = activities[
            ["activity_id", *[c for c in GLOBAL_METRIC_COLUMNS if c in activities.columns]]
        ]
        repo.upsert_activity_metrics(session, repo.records_to_dicts(globals_payload))

    return activities


# ═══════════════════════════════════════════════════════════════════════════
# KROK 2 – denní metriky
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate_daily_sums(daily: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """Denní součty per-activity metrik (metabolismus, tekutiny, EPOC)."""
    daily = daily.copy()
    if activities is None or activities.empty:
        return daily

    src = activities.copy()
    src["date"] = pd.to_datetime(src["date"])

    for col, out in (
        ("fat_kcal", "fat_kcal_daily"),
        ("carb_kcal", "carb_kcal_daily"),
        ("fat_g", "fat_g_daily"),
        ("carb_g", "carb_g_daily"),
        ("fluid_loss_l", "fluid_loss_l_daily"),
    ):
        if col in src.columns:
            agg = src.groupby("date")[col].sum()
            daily[out] = agg.reindex(daily.index)

    # max_hrr_60s se průměruje, ne sčítá – je to schopnost, ne objem.
    if "max_hrr_60s" in src.columns:
        valid = src[src["max_hrr_60s"].notna()]
        if not valid.empty:
            daily["max_hrr_60s_avg"] = valid.groupby("date")["max_hrr_60s"].mean().reindex(daily.index)
        else:
            daily["max_hrr_60s_avg"] = np.nan
    else:
        daily["max_hrr_60s_avg"] = np.nan

    return daily


def _compute_lthr(daily: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """
    Odhad prahového tepu z terénních dat.

    LTHR = 0.95 × nejlepší 20minutový průměr tepu v posledních 90 dnech.
    Na reálných datech dává 172 bpm, což na bpm sedí s laktátovým testem.

    Slouží jako REFERENCE, ne jako zdroj zón: měřená hodnota má přednost
    před odhadem. Ukazuje, kdy se práh posunul natolik, že stojí za to
    zóny v settings přenastavit.
    """
    daily = daily.copy()
    daily["lthr_estimate"] = np.nan

    col = f"best_{LTHR_TEST_MINUTES}min_hr"
    if activities is None or activities.empty or col not in activities.columns:
        return daily

    best = activities[["date", col]].dropna(subset=[col]).copy()
    if best.empty:
        return daily

    best["date"] = pd.to_datetime(best["date"])
    daily_best = best.groupby("date")[col].max().reindex(daily.index)
    rolling_best = daily_best.rolling(f"{LTHR_WINDOW_DAYS}D", min_periods=1).max()
    daily["lthr_estimate"] = (rolling_best * LTHR_FACTOR).round(0)
    return daily


def compute_daily_metrics(session: Session, activities: pd.DataFrame) -> pd.DataFrame:
    """
    Denní metriky nad celým kalendářem.

    Pořadí kroků kopíruje původní implementaci – navazují na sebe.
    """
    calendar = build_calendar(session)
    if len(calendar) == 0:
        return pd.DataFrame()

    biometrics_df = repo.read_biometrics_resolved(session)

    daily = load.build_daily_load(activities, calendar)
    daily = load.compute_ctl_atl_tsb(daily)
    daily = quality.compute_monotony_strain(daily)
    daily = quality.compute_efficiency_index(daily, activities)
    daily = _aggregate_daily_sums(daily, activities)
    daily = biometrics.compute_recovery(daily, biometrics_df)
    daily = biometrics.compute_readiness(daily)
    daily = biometrics.compute_illness_warning(daily)
    daily = load.compute_acwr(daily)
    daily = quality.compute_fatigue_index(daily)
    daily = advice.compute_coach_advice(daily)
    daily = quality.compute_polarization(daily, activities)
    daily = _compute_lthr(daily, activities)

    return daily.round({k: v for k, v in ROUND_MAP.items() if k in daily.columns})


def persist_daily_metrics(session: Session, daily: pd.DataFrame) -> int:
    """
    Zapíše denní metriky (upsert po dnech).

    `daily` vždy pokrývá celou osu z build_calendar, takže dny před jejím
    začátkem jsou pozůstatek starší, širší osy – upsert by je nechal ležet
    a export by je dál sypal do CSV.
    """
    if daily.empty:
        return 0

    cutoff = pd.to_datetime(daily.index.min()).date()
    stale = repo.delete_daily_metrics_before(session, cutoff)
    if stale:
        log.info("Smazáno %d dní před začátkem osy (%s).", stale, cutoff)

    out = daily[[c for c in DAILY_METRIC_COLUMNS if c in daily.columns]].copy()
    out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["metrics_version"] = DAILY_METRICS_VERSION
    out["computed_at"] = datetime.now()

    if "stress_flag_count" in out.columns:
        out["stress_flag_count"] = (
            pd.to_numeric(out["stress_flag_count"], errors="coerce").fillna(0).astype(int)
        )

    return repo.upsert_daily_metrics(session, repo.records_to_dicts(out))


# ═══════════════════════════════════════════════════════════════════════════
# VSTUPNÍ BOD
# ═══════════════════════════════════════════════════════════════════════════

def run_analytics(session: Session, force_activities: bool = False) -> AnalyticsResult:
    """Kompletní běh analytiky: per-activity metriky → denní metriky → zápis."""
    t0 = datetime.now()
    result = AnalyticsResult()

    activities = compute_activity_metrics(session, force=force_activities, result=result)
    daily = compute_daily_metrics(session, activities)

    if not daily.empty:
        result.days_written = persist_daily_metrics(session, daily)
        result.calendar_start = daily.index.min().date()
        result.calendar_end = daily.index.max().date()

    session.commit()
    result.duration_s = (datetime.now() - t0).total_seconds()
    log.info("Analytika hotová: %s", result.summary())
    return result
