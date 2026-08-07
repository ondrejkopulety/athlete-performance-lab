#!/usr/bin/env python3
"""
migrate_csv_to_db.py  –  jednorázový import historických CSV do databáze
========================================================================

Přenese existující data z data/summaries/*.csv do Postgresu, aby se
nemuselo znovu parsovat 864 FIT souborů a aby zůstala zachovaná historie.

    python scripts/migrate_csv_to_db.py              # vše
    python scripts/migrate_csv_to_db.py --skip-records   # bez vteřinových dat
    python scripts/migrate_csv_to_db.py --only biometrics

Zdroje → cíle:
    master_high_res_summary.csv        → activities + activity_metrics
    master_high_res_training_data.csv  → records            (COPY po chunkách)
    hrv / daily_health / sleep / vo2_max / movement / intensity
                                       → daily_biometrics   (sloučeno po dnech)
    athlete_readiness.csv              → daily_metrics      (baseline pro parity test)

CSV soubory zůstávají nedotčené – slouží dál jako záloha.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import SUMMARIES_DIR
from src.db import repository as repo
from src.db.session import session_scope
from src.ingestion.biometrics_import import import_biometrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate")

MASTER_SUMMARY = SUMMARIES_DIR / "master_high_res_summary.csv"
MASTER_RECORDS = SUMMARIES_DIR / "master_high_res_training_data.csv"
READINESS_CSV = SUMMARIES_DIR / "athlete_readiness.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Pomocné převody
# ═══════════════════════════════════════════════════════════════════════════

def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.date


def _to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})
    )


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1) activities + activity_metrics
# ═══════════════════════════════════════════════════════════════════════════

ACTIVITY_COLS = [
    "activity_id", "date", "activity_name", "sport", "duration_minutes",
    "total_trimp", "avg_hr", "max_hr", "time_in_z1", "time_in_z2", "time_in_z3",
    "time_in_z4", "time_in_z5", "zone2_cap_used", "records_count", "distance_km",
    "ascent_m", "descent_m", "avg_speed_kmh", "max_speed_kmh", "calories",
    "avg_cadence", "max_cadence", "avg_temp", "max_temp",
    "training_effect_aerobic", "training_effect_anaerobic", "vo2_max",
    "uphill_minutes", "source",
]

# CSV sloupec → sloupec v activity_metrics (rozdíly jen v případu písmen)
METRIC_COL_MAP = {
    "cardiac_drift": "cardiac_drift",
    "max_hrr_60s": "max_hrr_60s",
    "durability_pct": "durability_pct",
    "vam_m_per_h": "vam_m_per_h",
    "avg_gradient_pct": "avg_gradient_pct",
    "climb_category": "climb_category",
    "aet_hr_dfa": "aet_hr_dfa",
    "ant_hr_dfa": "ant_hr_dfa",
    "aet_hr_proxy": "aet_hr_proxy",
    "dfa_quality": "dfa_quality",
    "resp_rate_rsa": "resp_rate_rsa",
    "epoc_score": "epoc_score",
    "time_at_threshold_min": "time_at_threshold_min",
    "tte_z4z5_min": "tte_z4z5_min",
    "critical_hr": "critical_hr",
    "tati_score": "tati_score",
    "fat_kcal": "fat_kcal",
    "carb_kcal": "carb_kcal",
    "fat_g": "fat_g",
    "carb_g": "carb_g",
    "fluid_loss_L": "fluid_loss_l",
    "heat_flag": "heat_flag",
}


def migrate_activities(session) -> int:
    if not MASTER_SUMMARY.exists():
        log.warning("%s neexistuje – přeskakuji aktivity.", MASTER_SUMMARY)
        return 0

    df = pd.read_csv(MASTER_SUMMARY, dtype={"activity_id": str}, low_memory=False)
    log.info("master_high_res_summary.csv: %d řádků", len(df))

    df = df.drop_duplicates(subset=["activity_id"], keep="last")
    df["date"] = _to_date(df["date"])
    df = df.dropna(subset=["date"])

    numeric = [c for c in ACTIVITY_COLS if c not in ("activity_id", "date", "activity_name", "sport", "source")]
    df = _numeric(df, numeric)

    acts = df[[c for c in ACTIVITY_COLS if c in df.columns]].copy()
    acts["fit_path"] = None
    # fit_sha256 zůstává NULL → loader tyhle aktivity při dalším běhu ověří
    # proti FIT souborům na disku a doplní hash (případně přeparsuje).
    acts["fit_sha256"] = None
    n_acts = repo.upsert_activities(session, repo.records_to_dicts(acts))
    log.info("  → activities: %d řádků", n_acts)

    # Odvozené metriky: metrics_version=0 znamená „spočítáno starým kódem".
    # Pipeline je přepočítá, jakmile narazí na vyšší ACTIVITY_METRICS_VERSION.
    present = {csv_c: db_c for csv_c, db_c in METRIC_COL_MAP.items() if csv_c in df.columns}
    metrics = df[["activity_id", *present.keys()]].rename(columns=present)
    metrics = _numeric(metrics, [c for c in present.values() if c not in ("climb_category", "dfa_quality", "heat_flag")])
    if "heat_flag" in metrics.columns:
        metrics["heat_flag"] = _to_bool(metrics["heat_flag"])
    for int_col in ("aet_hr_dfa", "ant_hr_dfa", "aet_hr_proxy"):
        if int_col in metrics.columns:
            metrics[int_col] = metrics[int_col].round().astype("Int64")
    metrics["metrics_version"] = 0

    n_metrics = repo.upsert_activity_metrics(session, repo.records_to_dicts(metrics))
    log.info("  → activity_metrics: %d řádků (metrics_version=0)", n_metrics)
    return n_acts


# ═══════════════════════════════════════════════════════════════════════════
# 2) records – vteřinová data (556 MB → COPY po chunkách)
# ═══════════════════════════════════════════════════════════════════════════

def migrate_records(session, chunk_size: int = 250_000) -> int:
    if not MASTER_RECORDS.exists():
        log.warning("%s neexistuje – přeskakuji vteřinová data.", MASTER_RECORDS)
        return 0

    known_ids = set(repo.existing_activity_hashes(session).keys())
    log.info("Importuji vteřinová data (%.0f MB)…", MASTER_RECORDS.stat().st_size / 1e6)

    total = 0
    skipped_unknown = 0
    for i, chunk in enumerate(
        pd.read_csv(
            MASTER_RECORDS,
            dtype={"activity_id": str, "hr_zone": str},
            chunksize=chunk_size,
            low_memory=False,
        ),
        1,
    ):
        chunk = chunk[chunk["activity_id"].isin(known_ids)]
        before = len(chunk)
        chunk = chunk.dropna(subset=["timestamp"])
        skipped_unknown += before - len(chunk)
        if chunk.empty:
            continue

        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", format="mixed")
        chunk = chunk.dropna(subset=["timestamp"])
        # CSV nese timestampy s i bez offsetu → sjednotit na naivní UTC
        if getattr(chunk["timestamp"].dtype, "tz", None) is not None:
            chunk["timestamp"] = chunk["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        if "is_active" in chunk.columns:
            chunk["is_active"] = _to_bool(chunk["is_active"])
        chunk = _numeric(chunk, [c for c in repo.RECORD_COLUMNS
                                 if c not in ("activity_id", "timestamp", "hr_zone", "is_active")])

        rows = chunk[[c for c in repo.RECORD_COLUMNS if c in chunk.columns]].to_dict("records")
        total += repo.copy_records(session, rows)
        session.commit()   # ať se paměť transakce nehromadí přes 20 chunků
        log.info("  chunk %d: %d řádků (celkem %d)", i, len(rows), total)

    if skipped_unknown:
        log.info("  přeskočeno %d řádků bez platného timestampu", skipped_unknown)
    log.info("  → records: %d řádků", total)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# 3) daily_biometrics – sloučení šesti CSV po dnech
# ═══════════════════════════════════════════════════════════════════════════

def migrate_biometrics(session) -> int:
    """Sloučení šesti denních CSV řeší sdílený modul – stejnou cestou
    do DB vstupuje i biometrie z každodenního syncu."""
    log.info("Slučuji biometrické CSV…")
    return import_biometrics(session)


# ═══════════════════════════════════════════════════════════════════════════
# 4) daily_metrics – baseline pro parity test
# ═══════════════════════════════════════════════════════════════════════════

READINESS_RENAME = {
    "CTL": "ctl", "ATL": "atl", "TSB": "tsb",
    "fluid_loss_L_daily": "fluid_loss_l_daily",
}


def migrate_readiness(session) -> int:
    if not READINESS_CSV.exists():
        log.warning("%s neexistuje – přeskakuji baseline.", READINESS_CSV)
        return 0

    df = pd.read_csv(READINESS_CSV, low_memory=False)
    df["date"] = _to_date(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns=READINESS_RENAME)

    for bool_col in ("illness_warning", "ctl_ramp_warning"):
        if bool_col in df.columns:
            df[bool_col] = _to_bool(df[bool_col])

    text_cols = {"stress_flags", "coach_advice"}
    df = _numeric(df, [c for c in df.columns
                       if c not in text_cols | {"date", "illness_warning", "ctl_ramp_warning"}])
    df["metrics_version"] = 0

    n = repo.upsert_daily_metrics(session, repo.records_to_dicts(df))
    log.info("  → daily_metrics (baseline): %d dní", n)
    return n


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Import historických CSV do databáze")
    parser.add_argument(
        "--only",
        choices=["activities", "records", "biometrics", "readiness"],
        action="append",
        help="Importuj jen vybranou část (lze uvést vícekrát)",
    )
    parser.add_argument("--skip-records", action="store_true",
                        help="Přeskoč vteřinová data (nejdelší krok)")
    args = parser.parse_args()

    steps = args.only or ["activities", "records", "biometrics", "readiness"]
    if args.skip_records and "records" in steps:
        steps.remove("records")

    t0 = datetime.now()
    with session_scope() as session:
        if "activities" in steps:
            migrate_activities(session)
            session.commit()
        if "records" in steps:
            migrate_records(session)
        if "biometrics" in steps:
            migrate_biometrics(session)
            session.commit()
        if "readiness" in steps:
            migrate_readiness(session)
            session.commit()

        repo.set_state(session, "csv_migration", {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "steps": steps,
        })

    log.info("Hotovo za %.1f s", (datetime.now() - t0).total_seconds())


if __name__ == "__main__":
    main()
