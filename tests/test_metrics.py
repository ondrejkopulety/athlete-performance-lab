"""Fyziologické invarianty – hodnoty, které z definice nemohou vypadnout
z daného rozsahu. Chytí chybu ve vzorci dřív, než ji uvidí uživatel."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.quality import MONOTONY_CAP
from src.db import repository as repo


@pytest.fixture(scope="module")
def daily(request):
    from src.db.session import SessionLocal, check_connection

    if not check_connection():
        pytest.skip("Databáze neběží")
    s = SessionLocal()
    try:
        df = repo.read_daily_metrics(s)
    finally:
        s.close()
    if df.empty:
        pytest.skip("Žádné denní metriky – spusť `python scripts/main.py analyze`")
    return df


def _values(daily: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(daily[column], errors="coerce").dropna()


@pytest.mark.parametrize("column", ["readiness_score", "pure_recovery_score"])
def test_scores_are_percentages(daily, column):
    values = _values(daily, column)
    assert not values.empty
    assert values.min() >= 0
    assert values.max() <= 100


def test_whoop_strain_scale(daily):
    values = _values(daily, "whoop_strain")
    assert values.min() >= 0
    assert values.max() <= 21


def test_monotony_is_capped(daily):
    values = _values(daily, "monotony")
    assert values.max() <= MONOTONY_CAP


def test_polarization_shares_sum_to_100(daily):
    cols = ["polarization_low_pct", "polarization_high_pct", "z3_junk_pct"]
    subset = daily[cols].apply(pd.to_numeric, errors="coerce").dropna()
    assert not subset.empty
    total = subset.sum(axis=1)
    # 0.3 pp tolerance: každá složka se zaokrouhluje zvlášť na 1 desetinné místo
    assert (total - 100).abs().max() < 0.3


def test_tsb_equals_previous_day_balance(daily):
    df = daily.sort_values("date").reset_index(drop=True)
    ctl = pd.to_numeric(df["ctl"], errors="coerce")
    atl = pd.to_numeric(df["atl"], errors="coerce")
    tsb = pd.to_numeric(df["tsb"], errors="coerce")
    expected = (ctl.shift(1) - atl.shift(1)).round(2)
    both = expected.notna() & tsb.notna()
    assert (expected[both] - tsb[both]).abs().max() < 0.011  # tolerance zaokrouhlení


def test_sleep_need_within_model_bounds(daily):
    values = _values(daily, "sleep_need_min")
    assert values.min() >= 450   # základ 7.5 h
    assert values.max() <= 780   # strop 13 h


def test_rest_days_have_zero_load_not_null(daily):
    trimp = pd.to_numeric(daily["trimp"], errors="coerce")
    assert trimp.notna().all()
    assert (trimp >= 0).all()


def test_illness_warning_requires_multiple_flags(daily):
    from config.settings import ILLNESS_FLAG_COUNT

    flagged = daily[daily["illness_warning"] == True]  # noqa: E712
    if flagged.empty:
        return
    counts = pd.to_numeric(flagged["stress_flag_count"], errors="coerce")
    assert counts.min() >= ILLNESS_FLAG_COUNT
