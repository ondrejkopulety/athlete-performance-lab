"""
Shoda s původním výstupem
=========================

Refaktor nesmí tiše změnit čísla. Tenhle test porovnává daily_metrics
v databázi proti historickému athlete_readiness.csv, který vygeneroval
původní athlete_analytics.py.

Ze 37 společných sloupců jich 33 sedí na bit. Čtyři výjimky níže jsou
vědomé opravy – každá má vlastní test, aby se nedaly „opravit" zpátky
omylem.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import SUMMARIES_DIR
from src.db import repository as repo

BASELINE_CSV = SUMMARIES_DIR / "athlete_readiness.csv"

# CSV neslo jiná jména pro tři sloupce
RENAME = {"CTL": "ctl", "ATL": "atl", "TSB": "tsb", "fluid_loss_L_daily": "fluid_loss_l_daily"}

BOOL_COLUMNS = {"illness_warning", "ctl_ramp_warning"}
TEXT_COLUMNS = {"stress_flags", "coach_advice"}

# Sloupce s vědomou odchylkou – ověřují se samostatnými testy níže
KNOWN_DIFFERENCES = {
    "epoc_score_daily",           # dny volna: NULL → 0
    "recovery_tax_hours_daily",   # dtto
    "max_hrr_60s_avg",            # deduplikovaná vteřinová data
    "coach_advice",               # oprava překlepu „snič" → „sniž"
}


BASELINE_ACTIVITIES_CSV = SUMMARIES_DIR / "master_high_res_summary.csv"


def _comparable_until(session) -> "date | None":
    """
    Poslední den, který smí baseline popisovat.

    Každá aktivita, která přibyla po vygenerování baseline, legitimně mění
    metriky od svého data dál (rolling okna se dívají dozadu, takže starší
    dny zůstávají nedotčené). Bez téhle hranice by test začal selhávat po
    prvním syncu, přestože refaktor je v pořádku.
    """
    from datetime import timedelta

    if not BASELINE_ACTIVITIES_CSV.exists():
        return None
    baseline_ids = set(
        pd.read_csv(BASELINE_ACTIVITIES_CSV, dtype={"activity_id": str},
                    usecols=["activity_id"], low_memory=False)["activity_id"]
    )
    activities = repo.read_activities(session, with_metrics=False)
    if activities.empty:
        return None

    new = activities[~activities["activity_id"].isin(baseline_ids)]
    if new.empty:
        return None
    return min(new["date"]) - timedelta(days=1)


@pytest.fixture(scope="module")
def frames():
    from src.db.session import SessionLocal, check_connection

    if not BASELINE_CSV.exists():
        pytest.skip(f"Baseline {BASELINE_CSV} neexistuje")
    if not check_connection():
        pytest.skip("Databáze neběží")

    csv = pd.read_csv(BASELINE_CSV, low_memory=False)
    csv["date"] = pd.to_datetime(csv["date"]).dt.date
    csv = csv.rename(columns=RENAME).set_index("date")

    s = SessionLocal()
    try:
        db = repo.read_daily_metrics(s)
        cutoff = _comparable_until(s)
    finally:
        s.close()
    if db.empty:
        pytest.skip("Žádné denní metriky – spusť `python scripts/main.py analyze`")

    db["date"] = pd.to_datetime(db["date"]).dt.date
    db = db.set_index("date")

    common = csv.index.intersection(db.index)
    if cutoff is not None:
        common = common[common <= cutoff]
        if len(common) == 0:
            pytest.skip("Všechny dny baseline jsou ovlivněné novými aktivitami")
    return csv, db, common


def test_baseline_days_are_present_in_db(frames):
    """Každý den z baseline musí v databázi existovat (i dny bez tréninku)."""
    csv, db, common = frames
    assert len(common) > 0
    assert set(common) <= set(db.index)
    # Dny baseline, které nejsou v porovnání, smí chybět jen proto, že je
    # ovlivnila nová aktivita – ne proto, že by je databáze ztratila.
    missing = set(csv.index) - set(db.index)
    assert not missing, f"Databázi chybí dny z baseline: {sorted(missing)[:5]}"


@pytest.mark.parametrize(
    "column",
    sorted(
        {
            "ctl", "atl", "tsb", "acwr", "ctl_ramp_rate", "monotony", "strain",
            "whoop_strain", "daily_efficiency", "ef_trend", "fatigue_index",
            "readiness_score", "pure_recovery_score", "hrv_last_night",
            "hrv_weekly_avg", "hrv_cv_pct", "rhr_day", "avg_stress_day",
            "sleep_score_day", "sleep_duration_min", "sleep_need_min",
            "sleep_performance_pct", "polarization_low_pct",
            "polarization_high_pct", "stress_flag_count",
            "fat_kcal_daily", "carb_kcal_daily", "fat_g_daily", "carb_g_daily",
            "fluid_loss_l_daily",
        }
    ),
)
def test_numeric_column_matches_baseline(frames, column):
    csv, db, common = frames
    assert column not in KNOWN_DIFFERENCES

    a = pd.to_numeric(csv.loc[common, column], errors="coerce")
    b = pd.to_numeric(db.loc[common, column], errors="coerce")

    assert (a.notna() == b.notna()).all(), f"{column}: liší se přítomnost hodnot"
    both = a.notna() & b.notna()
    assert (a[both] - b[both]).abs().max() < 1e-6, f"{column}: číselná odchylka"


@pytest.mark.parametrize("column", sorted(BOOL_COLUMNS))
def test_boolean_column_matches_baseline(frames, column):
    csv, db, common = frames
    a = csv.loc[common, column].astype(str).str.lower().isin(["true", "1", "1.0"])
    b = db.loc[common, column].fillna(False).astype(bool)
    assert (a == b).all()


def test_stress_flags_match_baseline(frames):
    csv, db, common = frames
    a = csv.loc[common, "stress_flags"].fillna("").astype(str)
    b = db.loc[common, "stress_flags"].fillna("").astype(str)
    assert (a == b).all()


# ── Vědomé odchylky ────────────────────────────────────────────────────────

@pytest.mark.parametrize("column", ["epoc_score_daily", "recovery_tax_hours_daily"])
def test_rest_days_now_zero_instead_of_null(frames, column):
    """
    Původní kód nechával dny volna prázdné. Nula je pravdivější (netrénoval
    jsem → žádné EPOC) a odpovídá tomu, jak se chová trimp.

    Tam, kde hodnota existovala v obou, musí být identická.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, column], errors="coerce")
    b = pd.to_numeric(db.loc[common, column], errors="coerce")

    both = a.notna() & b.notna()
    assert (a[both] - b[both]).abs().max() < 1e-6

    only_db = a.isna() & b.notna()
    assert (b[only_db] == 0).all(), "Doplněné hodnoty musí být nuly, ne jiná čísla"
    assert not (a.notna() & b.isna()).any(), "Nesmíme ztratit hodnotu, kterou baseline měl"


def test_coach_advice_differs_only_by_typo_fix(frames):
    """Jediný rozdíl v textu doporučení je oprava překlepu „snič" → „sniž"."""
    csv, db, common = frames
    a = csv.loc[common, "coach_advice"].fillna("").astype(str)
    b = db.loc[common, "coach_advice"].fillna("").astype(str)

    differing = a[a != b]
    assert not differing.empty, "Očekáváme právě opravu překlepu"
    assert (a.str.replace("snič intenzitu", "sniž intenzitu", regex=False) == b).all()


def test_max_hrr_differences_are_bounded(frames):
    """
    max_hrr_60s_avg se liší jen tam, kde historické CSV obsahovalo tutéž
    aktivitu naparsovanou dvakrát. Databáze drží deduplikovaná data, takže
    jde o opravu – rozdíl musí být malý a týkat se hrstky dní.
    """
    csv, db, common = frames
    a = pd.to_numeric(csv.loc[common, "max_hrr_60s_avg"], errors="coerce")
    b = pd.to_numeric(db.loc[common, "max_hrr_60s_avg"], errors="coerce")

    both = a.notna() & b.notna()
    diff = (a[both] - b[both]).abs()
    assert (diff > 1e-6).sum() <= 5, "Odchylek je víc, než odpovídá duplicitám v CSV"
    assert diff.max() < 5.0, "Odchylka je příliš velká na chybu z duplicit"
